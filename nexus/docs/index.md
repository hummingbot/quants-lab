# Strategy parameters
(parsed in abc.py, class Strategy(ABC))

#### symbol_list

#### look_back
Number of bars that are executed before strategy can begin to trade (default = 80). Required for most indicators or other functions that need a certain amount of previous price history for their calculations. If left at the default value, it is automatically adapted to the needed lookback period by subsequent indicators[**THIS FEATURE TODO TO IMPLENT**]. Otherwise set it to a value that is sure to cover the longest period of all used indicators, assets, and time frames. Set it to 0 when the script needs no lookback period. Backtests can have any lookback period as long as it does not exceed the total number of bars. 

#### max_long, max_short
Maximum number of open and pending long or short trades with the same asset and algo. If the limit amount is reached mode, enter calls do not enter more trades, but they still close reverse positions and update the stop limits, profit targets, and life times of open trades to the values that the new trade would have. If set to a negative number, open trades are not updated.

#### lots
Trade size given by an integer number of lots (default = 1; max = **lot_lmit**). 1 lot is defined as the smallest possible order size of the selected asset. Thus, a lot can be a multiple or a fraction of a contract, determined by LotAmount. Since it's the minimum, the trade size can never be less than 1 lot (see remarks).

Lot has different meanings dependent on platform and broker. Normally, 1 lot is the smallest order unit. The trade size is always an integer number of lots, and there is no such thing as a 'fractional' or 'partial' lot. The lot amount - the number of contracts or units equivalent to one lot, available through the LotAmount variable - depends on the broker, the account, and the asset type. Forex brokers can offer mini lot and micro lot accounts. 1 mini lot is equivalent to 10,000 contracts and about \$100 margin; 1 micro lot is 1000 contracts and about \$10 margin. On stock broker accounts, the lot amounts and margins for forex trading are usually higher and the leverage is smaller. For CFDs, some brokers offer lot sizes that are a fraction of one contract (f.i.1 lot = 0.1 contracts). 

#### lot_amount
The number of contracts per lot with the current asset. Determines the minimum order size and depends on the lot size of the account. For currencies, the lot size of a micro lot account is normally 1000 contracts; of a mini lot account 10000 contracts; and of a a standard lot account 100000 contracts. Some brokers offer also lot sizes that are a fraction of a contract, f.i. 0.1 contracts per lot for CFDs or diginal coins.

#### lot_lmit
Maximum number of **lots** with the current asset (default: 1000000000/**lot_amount**). Can be set to a smaller number for safety reasons.