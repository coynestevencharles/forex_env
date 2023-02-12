# ForexTradingEnv
A flexible environment for currency trading with reinforcement learning. Follows the gym interface.

This environment includes counts of the currencies involved, a difference from the more abstract gym-anytrading. 

The currencies are also set as the USD-JPY pair by default, but this can easily be changed (simply replacing variable names and print text should suffice).

This environment can be used with reinforcement learning such as those found in Stable Baselines 3. The "action_space_type" parameter affects which algorithm to use.

This environment is compatible with "ohlc" data, such as that provided by yfinance and many other apis, as well as more detailed data with both ask and bid prices. See the "df_type" argument below.
