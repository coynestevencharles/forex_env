import numpy as np
import pandas as pd
import gym
from gym import spaces
from matplotlib import pyplot as plt

BUY = 0
SELL = 1
HOLD = 2

class ForexTradingEnv(gym.Env):
    """
    Flexible environment for Forex Trading that follows gym interface
    """
    metadata = {'render.modes': ['human']}
    visualization = None

    def __init__(self, df: pd.DataFrame, action_space_type: str, window_size: int, 
                 frame_bound, df_type: str='ohlc', initial_jpy_amount: int = 1000000) -> None:
        super(ForexTradingEnv, self).__init__()

        assert len(frame_bound) == 2

        self.df = df.copy()
        self.action_space_type = action_space_type
        self.df_type = df_type
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.initial_jpy_amount = initial_jpy_amount
        self._first_rendering = None
        self._done = False
        self._trades = []
        self._profit = 0
        self._current_step = self.frame_bound[0] - self.window_size

        if self.action_space_type == 'discrete':
            # Actions that we can take: BUY, SELL, HOLD
            self.action_space = spaces.Discrete(3)
        elif self.action_space_type == 'percentage':
            # Actions of the format: BUY x%, SELL x%, HOLD
            self.action_space = spaces.Box(
                low=np.array([0, 0], dtype=np.float32), high=np.array([2, 1], dtype=np.float32), dtype=np.float32
            )
        else:
            raise ValueError(f'Bad argument to action_space_type: {action_space_type}')

        # Prices contains OHLC values, perhaps both ask and bid prices, for last N days
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.df.shape[1], self.window_size+1), dtype=np.float32
        )

        # Create a DF containing mean price values for plotting purposes
        if self.df_type == 'ask_bid':
            self._mid_df = self.df.copy()
            self._mid_df['open'] = self._mid_df[['ask_open', 'bid_open']].mean(axis=1)
            self._mid_df['high'] = self._mid_df[['ask_high', 'bid_high']].mean(axis=1)
            self._mid_df['low'] = self._mid_df[['ask_low', 'bid_low']].mean(axis=1)
            self._mid_df['close'] = self._mid_df[['ask_close', 'bid_close']].mean(axis=1)

            self._mid_df = self._mid_df[['open', 'high', 'low', 'close']]

    def _next_observation(self):
        frame = np.zeros((self.df.shape[1], self.window_size+1))

        sliced_df = self.df.iloc[self._current_step:self._current_step+self.window_size]

        if self.df_type == 'ohlc':
            price_col_count = 4
        elif self.df_type == 'ask_bid':
            price_col_count = 8
        else:
            raise ValueError('df_type must be "ohlc" or "ask_bid"')
        
        custom_signals = []
        price_signals = [(sliced_df[col].values) for col in sliced_df.iloc[:,:price_col_count]]
        if sliced_df.shape[1] > price_col_count:
            custom_signals = [sliced_df[col].values for col in sliced_df.iloc[:,price_col_count:]]
        np.put(frame, [0, sliced_df.shape[1]], [price_signals + custom_signals])
        
        return frame

    def _take_action(self, action):
        if self.df_type == 'ask_bid':
            current_ask_price = np.random.uniform(
                self.df.iloc[self._current_step]['ask_open'], self.df.iloc[self._current_step]['ask_close'])
            current_bid_price = np.random.uniform(
                self.df.iloc[self._current_step]['bid_open'], self.df.iloc[self._current_step]['bid_close'])
        elif self.df_type == 'ohlc':
            current_ask_price = np.random.uniform(
                self.df.iloc[self._current_step]['open'], self.df.iloc[self._current_step]['close'])
            current_bid_price = np.random.uniform(
                self.df.iloc[self._current_step]['open'], current_ask_price)
        else:
            raise ValueError('df_type must be "ohlc" or "ask_bid"') 
        

        if self.action_space_type == 'percentage':
            action_type = round(action[0])
            amount = action[1]
        elif self.action_space_type == 'discrete':
            action_type = action
            amount = 1

        if action_type == BUY:
            # Convert % amount of balance into USD
            total_possible = self._balance / current_ask_price
            usd_bought = total_possible * amount
            cost_to_buy = (usd_bought * current_ask_price) + ((usd_bought * current_ask_price) * 0.0003)

            self._balance -= cost_to_buy
            self._usd_held += usd_bought

            self._trades.append({
                'step': self._current_step,
                'usd': usd_bought,
                'cost': cost_to_buy,
                'trade_type': 'buy'
            })

        elif action_type == SELL:
            # Sell % amount of USD 
            usd_sold = self._usd_held * amount
            cost_to_sell = usd_sold * current_bid_price - ((usd_sold * current_bid_price) * 0.0003)
            self._balance += cost_to_sell
            self._usd_held -= usd_sold

            self._trades.append({
                'step': self._current_step,
                'usd': usd_sold,
                'cost': cost_to_sell,
                'trade_type': 'sell'
            })         

        elif action_type == HOLD:
            self._trades.append({
                'step': self._current_step,
                'trade_type': 'hold'
            })

        self._net_worth = self._balance + (self._usd_held * current_bid_price)
        self._profit = self._net_worth - self.initial_jpy_amount
        if self._net_worth > self._max_net_worth: 
            self._max_net_worth = self._net_worth

    def step(self, action):
        step_start_amount = self._net_worth
        self._take_action(action)
        self._current_step += 1

        reward = self._net_worth - step_start_amount # Reward = stepwise profit
        done = self._net_worth <= 0 or self._current_step >= self.frame_bound[1]
        self._done = done

        info = dict(
            step_reward = reward,
            net_worth = self._net_worth,
            trade = action
        )

        obs = self._next_observation()

        return obs, reward, done, info

    def reset(self):
        
        self._balance = self.initial_jpy_amount
        self._net_worth = self.initial_jpy_amount # In JPY
        self._max_net_worth = self.initial_jpy_amount # In JPY
        self._usd_held = 0
        self._current_step = self.frame_bound[0] - self.window_size
        self._trades = []
        self._first_rendering = True
        self._done = False
        self._profit = 0

        return self._next_observation()
    
    def print_status(self):
        print(f'Yen Held: {self._balance:.2f}; USD held: {self._usd_held:.2f}')
        print(f'Net worth: {self._net_worth:.2f} (Max net worth: {self._max_net_worth:.2f})')
        print(f'Profit: {self._profit:.2f} yen.')

    def render(self, mode='human'):

        if self.df_type == 'ask_bid':
            plot_df = self._mid_df
        elif self.df_type == 'ohlc':
            plot_df = self.df

        buy_steps = []
        sell_steps = []
        hold_steps = []

        for i in range(len(self._trades)):
            if self._trades[i]['trade_type'] == 'buy':
                buy_steps.append(self._trades[i]['step'])
            elif self._trades[i]['trade_type'] == 'sell':
                sell_steps.append(self._trades[i]['step'])
            elif self._trades[i]['trade_type'] == 'hold':
                hold_steps.append(self._trades[i]['step'])

        fig, ax = plt.subplots(figsize=(16,9))

        ax.plot(plot_df['close'])
        ax.plot(plot_df.index.values[buy_steps], plot_df.iloc[buy_steps, :]['close'], 'go', label='Buy')
        ax.plot(plot_df.index.values[sell_steps], plot_df.iloc[sell_steps, :]['close'], 'ro', label='Sell')
        
        ax.set_xlim(plot_df.index[self.frame_bound[0]], plot_df.index[self.frame_bound[1]-1])
        ax.set_ylim(plot_df['close'].iloc[self.frame_bound[0]:self.frame_bound[1]].min() * 0.95,
                    plot_df['close'].iloc[self.frame_bound[0]:self.frame_bound[1]].max() * 1.05)
        
        ax.legend()
        plt.suptitle(
        f'Yen Held: {self._balance:.2f}; USD held: {self._usd_held:.2f}' +
        f'\nNet worth: {self._net_worth:.2f} (Max net worth: {self._max_net_worth:.2f})' +
        f'\nProfit: {self._profit:.2f} yen.'
        )
        plt.show()