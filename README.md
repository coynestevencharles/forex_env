# Forex Environment for Reinforment Learning

Contains ForexTradingEnv, a flexible environment for currency trading with reinforcement learning. Follows the [OpenAI gym](https://www.gymlibrary.dev/) interface.

The code for this project was based on [gym-anytrading](https://github.com/AminHP/gym-anytrading) and [Stock-Trading-Environment](https://github.com/notadamking/Stock-Trading-Environment).

This environment can be used with reinforcement learning such as those found in [Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/).

This environment is compatible with "ohlc" data, such as that provided by [yfinance](https://github.com/ranaroussi/yfinance) and many other apis, as well as more detailed data with both ask and bid prices. See the `df_type` parameter below.

This environment includes counts of the currencies involved, a difference from the more abstract gym-anytrading. The currencies are also set as the USD-JPY pair by default, but this can easily be changed (simply replacing variable names and print text should suffice).

## ForexTradingEnv

Properties:

> `df`: A pandas dataframe containing the information that the model will act on. It is passed in the constructor.
> `df_type`: Either "ohlc" or "ask_bid", passed in the class constructor. This describes the price data in `df`.
> * "ohlc" assumes exactly four price columns, named "open", "high", "low", and "close". This is similar to the format provided by yfinance and many other apis.
> * "ask_bid" assumes exactly eight price columns, named "ask_open", "ask_high", "ask_low", "ask_close", "bid_open", "bid_high", "bid_low", and "bid_close", and is suitable when you have separate ask and bid price data.

> `window_size`: Number of ticks (current and previous ticks) returned as a *Gym observation*. It is passed in the constructor.

> `action_space_type`: Either "discrete" or "percentage", passed in the constructor. This affects the `action space` property and determines which reinforcement learning models can be used with the environment (see [here](https://stable-baselines3.readthedocs.io/en/master/guide/algos.html)). The "percentage" option corresponds to "Box".

> `action_space`: The *Gym action_space* property. Containing discrete values of **0=Sell**, **1=Buy**, and **2=Hold** when `action_space_type` is "discrete" and a *Box* expressing the above choices as well as a range from 0 to 1 when `action_space_type` is "percentage". The range represents the proportion of held currency to trade. Note that trades in "discrete" are always full exchanges of all held currency.

> `frame_bound`: A tuple which specifies the start and end of `df`. It is passed in the constructor. The first element should be equal to or greater than `window_size`.

> `observation_space`: The *Gym observation_space* property. Each observation is a window on `df` from index **current_step - window_size + 1** to **current_step**. The starting tick of the environment is thus equal to `window_size`. 

> `position_history`: Stores the information of all steps.

* Methods:
> `reset`: Typical *Gym reset* method.
>
> `step`: Typical *Gym step* method.
>
> `render`: Renders the information of the environment's current tick and displays a chart of buy and sell actions taken.

## Examples

### Creating an Environment

```python
import gym
import pandas as pd
from forex_env import ForexTradingEnv

df = pd.read_csv(file_name) # Additional processing likely necessary

env = ForexTradingEnv(df=df, action_space_type='discrete', df_type='ohlc', window_size=5, frame_bound=(5, len(df)))
```

### Testing or Making Predictions

This assumes you have trained a model and can call model.predict().

If `action_space_type` is "discrete":

```python
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action)
    if done:
        break

env.render()
```

If `action_space_type` is "percentage":

```python
while True: 
    obs = obs[np.newaxis, ...]
    action, _states = model.predict(obs)
    obs, rewards, done, info = env.step(action[0]) # Note the index of [0]
    if done:
        break

env.render()
```

### Full Example with yfinance and Stable Baselines 3

See the notebook [here](https://github.com/coynestevencharles/forex_env/blob/main/example_yf_sb3.ipynb) for an example using these libraries.

### Adding Additional Features to the Observation Space ("Signal Features")

Any additional columns found in the `df` are considered signal features. You can add e.g. news sentiment columns, moving averages, or other financial indicators. Therefore, please make sure that the dataframe passed to the model contains only the 4 (or 8) price columns plus the data you intend as signal features.

The code for the price and custom signal columns is separate, so you can treat them differently by editing the `_next_observation` method. This would allow one to e.g. internally scale price values only.
