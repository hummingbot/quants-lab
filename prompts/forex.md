
# This is repo of an python based algorithmic trading event driven framework code name NEXUS.

This framework supports multiple asset classes, including forex, futures, stocks, ETFs, and cryptocurrencies.
Different asset classes has different rules for trading and different set of assets parameters.

The data for forex asset located in file: nexus\data\zorro\assets.csv
```
Name,Price,Spread,RollLong,RollShort,PIP,PIPCost,MarginCost,Market,LotAmount,Commission,Symbol
AUD/USD,0.63663,0.00016,0.24,-0.51,0.0001,0.09197,-6.66,0,1000,0.6,AUD/USD
EUR/CHF,1.05201,0.00016,0.08,-0.5294,0.0001,0.095084,-3.33,0,1000,0.8,EUR/CHF
EUR/USD,1.08762,0.00012,-0.3,0.034,0.0001,0.09197,-3.33,0,1000,0.6,EUR/USD
GBP/USD,1.25082,0.00011,0.05,-0.12,0.0001,0.09197,-3.33,0,1000,0.6,GBP/USD
```
The asset parameters are to be set up as described below:

**Name** - Name of the asset, f.i. "EUR/USD". Up to 15 characters, case sensitive, with no blanks and no special characters except for slash '/' and underline '_'.Name of the asset, f.i. "EUR/USD". Up to 15 characters, case sensitive, with no blanks and no special characters except for slash '/' and underline '_'
**Price** - Ask price per unit, in counter currency units. Accessible with the InitialPrice variable.
**Spread** - The current difference of ask and bid price, in counter currency units.Used for backtests with constant spread. Accessible with the Spread variable.
**RollLong**, **RollShort** - Rollover fee (Swap) in account currency units for holding overnight a long or short position per calendar day and per 10000 units for currencies, or per unit for all other assets. Accessible with the RollLong/Short variables.
**PIP Size**-  of 1 pip in counter currency units. Equivalent to the traditional smallest increment of the price. The usual pip size is 0.0001 for forex pairs with a single digit price, 0.01 for stocks. accessible with the PIP variable.
**PipCost** - Value of 1 pip profit or loss per lot, in units of the account currency. Accessible with the PipCost variable and internally used for calculating trade profits or losses. When the asset price rises or falls by x, the equivalent profit or loss in account currency is x * Lots * PIPCost / PIP. For assets with pip size 1 and one contract per lot, the pip cost is the conversion factor from counter currency to account currency. For calculating it manually, multiply LotAmount with PIP; if the counter currency is different to the account currency, multiply the result with the counter currency exchange rate. Example 1: AUD/USD on a micro lot EUR account has PipCost of 1000 * 0.0001 * 0.9 (current USD price in EUR) = 0.09 EUR. Example 2: AAPL stock on a USD account has PipCost of 1 * 0.01 = 0.01 USD = 1 cent. Example 3: S&P500 E-Mini futures (ES) on a USD account have PipCost of 12.5 USD (1 point = 25 cent price change of the underlying is equivalent to $12.50 profit/loss of an ES contract). This paramter is affected by LotAmount.
**MarginCost** - Margin required for purchasing and holding 1 lot of the asset, either in units of the account currency or in percent of the position value. Depends on account leverage, account currency, counter currency, and LotAmount. If the broker has different values for initial, maintenance, and end-of-day margin, use the highest of them. Accessible with the MarginCost variables. Internally used for the conversion from trade Margin to Lot amount: the number of lots that can be purchased with a given trade margin is Margin / MarginCost. Also affects the Required Capital and the Annual Return in the performance report. 
**Market** -  Time zone and market hours in the format ZZZ:HHMM-HHMM, for instance EST:0930-1545. Available to the script in the variables AssetMarketZone, AssetMarketStart, AssetMarketEnd, for skipping data series or preventing orders outside market hours.
**LotAmount** - Number of units for 1 lot of the asset; accessible with the LotAmount variable. Smallest amount that you can buy or sell with your broker without getting the order rejected or an "odd lot size" warning. For forex the LotAmount is normally 1000 on a micro lot account, 10000 on a mini lot account, and 100000 on standard lot accounts. Index CFDs or crypto currencies can have a LotAmount less than 1, such as 0.1 CFD contracts. For stocks and most other assets the lot amount is normally 1. The lot amount affects PipCost and MarginCost.
   Cryptocurrency broker APIs often support the SET_AMOUNT command. You can then set any desired lot amount in this field, such as 0.0001 Bitcoins. 10000 Lots are then equivalent to 1 Bitcoin. 
   A negative value in this field is interpreted as a multiplier for option or future contracts, and accessible with the Multiplier variable. The LotAmount is then set to 1. 
**Commission** Roundturn commission amount in account currency units for opening and closing one contract, per 10000 units for currencies, per unit for other assets, per underlying unit for options, or as percent of the trade value when negative. Accessible with the Commission variable. When manually entering the commission, double it if was single turn. For forex pairs make sure to convert it to per-10000. If an asset is a forex pair can be determined with the assetType function. A forex commission in pips is converted by Commission = CommissionInPips * 10000 * PIP.
  If the commission is a percentage of the trade value, enter the negative percent value, f.i. -0.25 for 0.25%. A percentage can be converted to commission amount with the formula Commission = -Percent/100 * Price / LotAmount * PIPCost / PIP.
**Symbol** Broker symbol(s) of the asset. Up to 3 sources and symbols can be entered for trading, receiving live prices, and receiving historical prices, separated by '!' exclamation marks in the form trade!live!history. Leave this field empty or enter '*' when the symbol is identical to the asset name or is automatically converted by the broker plugin. Up to 128 characters, case sensitive.

# Refactor code base:
Add asset types  as ‘forex’, ‘crypto’, ‘stock’, ‘future’, etc..
Read file nexus\data\zorro\assets.csv with (add to it to  backtest spec)
Implement trading rules for ‘forex’  asset type.

