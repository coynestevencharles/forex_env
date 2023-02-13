import numpy as np
import pandas as pd
import gym
from gym import spaces
import random
from matplotlib import pyplot as plt

INITIAL_JPY_AMOUNT_HAD = 1000000

BUY = 0
SELL = 1
HOLD = 2

class ForexTradingEnv(gym.Env):
    """
    Flexible environment for Forex Trading that follows gym interface
    """
    metadata = {'render.modes': ['human']}
    visualization = None

    def __init__(self, df: pd.DataFrame, action_space_type: str, window_size: int, frame_bound, df_type: str='ohlc') -> None:
        super(ForexTradingEnv, self).__init__()

        assert len(frame_bound) == 2

        self.df = df.copy()
        self.action_space_type = action_space_type
        self.df_type = df_type
        self.window_size = window_size
        self.frame_bound = frame_bound
        self.done = False
        self.position_history = []

        if self.action_space_type == 'discrete':
            # Actions that we can take: BUY, SELL, HOLD
            self.action_space = spaces.Discrete(3)
        elif self.action_space_type == 'percentage':
            # Actions of the format: BUY x%, SELL x%, HOLD
            self.action_space = spaces.Box(
                low=np.array([0, 0], dtype=np.float32), high=np.array([2, 1], dtype=np.float32), dtype=np.float32
            )
        else:
            raise ValueError('Bad argument to action_space_type', action_space_type) 

        # Prices contains OHLC values, perhaps of both ask and bid prices, for the last N days
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.df.shape[1], self.window_size+1), dtype=np.float32
        )

    def _next_observation(self):
        frame = np.zeros((self.df.shape[1], self.window_size+1))

        sliced_df = self.df.iloc[self.current_step:self.current_step+self.window_size]
        price_col_count = 4 if self.df_type == 'ohlc' else 8 if self.df_type == 'ask_bid' else None
        
        custom_signals = []
        price_signals = [(sliced_df[col].values) for col in sliced_df.iloc[:,:price_col_count]]
        if sliced_df.shape[1] > price_col_count:
            custom_signals = [sliced_df[col].values for col in sliced_df.iloc[:,price_col_count:]]
        np.put(frame, [0, sliced_df.shape[1]], [price_signals + custom_signals])
        
        # signals = [(sliced_df[col].values) for col in sliced_df]
        # np.put(frame, [0, sliced_df.shape[1]], signals)
        
        return frame

    def _take_action(self, action):
        if self.df_type == 'ask_bid':
            current_ask_price = random.uniform(
                self.df.iloc[self.current_step]['ask_open'], self.df.iloc[self.current_step]['ask_close'])
            current_bid_price = random.uniform(
                self.df.iloc[self.current_step]['bid_open'], self.df.iloc[self.current_step]['bid_close'])
        elif self.df_type == 'ohlc':
            current_ask_price = random.uniform(
                self.df.iloc[self.current_step]['open'], self.df.iloc[self.current_step]['close'])
            current_bid_price = random.uniform(
                self.df.iloc[self.current_step]['open'], current_ask_price)
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
            total_possible = self.balance / current_ask_price
            usd_bought = total_possible * amount
            cost_to_buy = usd_bought * current_ask_price + ((usd_bought * current_ask_price) * 0.0003)

            self.balance -= cost_to_buy
            self.usd_held += usd_bought

            if usd_bought > 0:
                self.trades.append({
                    'step': self.current_step,
                    'usd': usd_bought,
                    'cost': cost_to_buy,
                    'type': 'buy'
                })

        elif action_type == SELL:
            # Sell % amount of USD 
            usd_sold = self.usd_held * amount
            cost_to_sell = usd_sold * current_bid_price - ((usd_sold * current_bid_price) * 0.0003)
            self.balance += cost_to_sell
            self.usd_held -= usd_sold

            if usd_sold > 0:
                self.trades.append({
                    'step': self.current_step,
                    'usd': usd_sold,
                    'cost': cost_to_sell,
                    'type': 'sell'
                })         

        elif action_type == HOLD:
            self.trades.append({
                'step': self.current_step,
                'type': 'hold'
            })

        self.net_worth = self.balance + self.usd_held * current_bid_price
        if self.net_worth > self.max_net_worth: 
            self.max_net_worth = self.net_worth

        self.position_history = self.trades

    def step(self, action):
        step_start_amount = self.net_worth
        self._take_action(action)
        self.current_step += 1

        reward = self.net_worth - step_start_amount # Reward = stepwise profit
        done = self.net_worth <= 0 or self.current_step >= self.frame_bound[1]
        self.done = done

        info = dict(
            step_reward = reward,
            net_worth = self.net_worth,
            position = action
        )

        obs = self._next_observation()

        return obs, reward, done, info

    def reset(self):
        self.balance = INITIAL_JPY_AMOUNT_HAD
        self.net_worth = INITIAL_JPY_AMOUNT_HAD # In JPY
        self.max_net_worth = INITIAL_JPY_AMOUNT_HAD # In JPY
        self.usd_held = 0
        self.current_step = 0
        self.trades = []
        self.position_history = []
        self.current_step = self.frame_bound[0] - self.window_size

        return self._next_observation()

    def render(self, mode='human'):
        profit = self.net_worth - INITIAL_JPY_AMOUNT_HAD
        
        print(f'Yen Held: {self.balance:.2f}; USD held: {self.usd_held:.2f}')
        print(f'Net worth: {self.net_worth:.2f} (Max net worth: {self.max_net_worth:.2f})')
        print(f'Profit: {profit:.2f} yen.')

        # Plot the mid (i.e. between ask and bid) prices line

        if self.done:
            
            self.mid_df = self.df.copy()
            
            if self.df_type == 'ask_bid':
                self.mid_df['mid_open'] = self.mid_df[['ask_open', 'bid_open']].mean(axis=1)
                self.mid_df['mid_high'] = self.mid_df[['ask_high', 'bid_high']].mean(axis=1)
                self.mid_df['mid_low'] = self.mid_df[['ask_low', 'bid_low']].mean(axis=1)
                self.mid_df['mid_close'] = self.mid_df[['ask_close', 'bid_close']].mean(axis=1)

                self.mid_df = self.mid_df[['mid_open', 'mid_high', 'mid_low', 'mid_close']]
                self.mid_df = self.mid_df.rename(columns={
                    'mid_open': 'open',
                    'mid_high': 'high',
                    'mid_low': 'low',
                    'mid_close': 'close',
                })

            # Plot ticks on the line

            buy_ticks = []
            sell_ticks = []
            hold_ticks = []

            for i in range(len(self.position_history)):
                if self.position_history[i]['type'] == 'buy':
                    buy_ticks.append(self.position_history[i]['step'])
                elif self.position_history[i]['type'] == 'sell':
                    sell_ticks.append(self.position_history[i]['step'])
                elif self.position_history[i]['type'] == 'hold':
                    hold_ticks.append(self.position_history[i]['step'])

            fig, ax = plt.subplots(figsize=(16,9))

            ax.plot(self.mid_df['close'])
            ax.plot(self.mid_df.index.values[buy_ticks], self.mid_df.iloc[buy_ticks, :]['close'], 'go', label='Buy')
            ax.plot(self.mid_df.index.values[sell_ticks], self.mid_df.iloc[sell_ticks, :]['close'], 'ro', label='Sell')
            # If you want to see the hold actions, uncomment the following line
            #ax.plot(self.mid_df.index.values[hold_ticks], self.mid_df.iloc[hold_ticks, :]['close'], 'yo', label='Hold')
            
            ax.set_xlim(self.mid_df.index[self.frame_bound[0]], self.mid_df.index[self.frame_bound[1]-1])
            ax.set_ylim(self.mid_df['close'].iloc[self.frame_bound[0]:self.frame_bound[1]].min()*0.95,
                        self.mid_df['close'].iloc[self.frame_bound[0]:self.frame_bound[1]].max()*1.05)
            
            ax.legend()
            plt.show()