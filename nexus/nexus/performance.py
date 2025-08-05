from nexus.__version__ import __version__
import logging
import numpy as np
import pandas as pd
import os
class PerformanceReport:
    """Generates performance reports from trade data."""

    def __init__(self, data_handler, portfolio, execution_time=None):
        self.logger = logging.getLogger(self.__class__.__name__) 
        self.data_handler = data_handler
        self.portfolio = portfolio
        self.execution_time = execution_time 
        self.equity = self.create_equity_curve()  # Total for overall portfolio
        self.trades = self.construct_all_trades()  # Reconstruct trades with profits and returns
       
    def create_equity_curve(self):
        equity = pd.DataFrame(self.portfolio.all_holdings)
        equity.set_index('datetime', inplace=True)
        equity.index = pd.to_datetime(equity.index)
        # Remove rows where index is NaT
        equity = equity[~equity.index.isnull()]
        equity['returns'] = equity['total'].pct_change().fillna(0.0)
        return equity

    def calculate_symbol_equity_curve(self, symbol):
        # Filter the trades for the given symbol
        filtered_trades = self.trades[self.trades['symbol'] == symbol]
        
        # Create a new DataFrame with the exit_datetime as the index and cumulative sum of profit as the equity curve
        equity = pd.DataFrame({
            'timestamp': filtered_trades['exit_datetime'],
            'equity': filtered_trades['profit'].cumsum()
        })
        
        # Set the timestamp as the index for easier plotting or analysis
        equity.set_index('timestamp', inplace=True)
        return equity

    def construct_all_trades(self):
        """Constructs a list of trades with entry and exit points for all symbols."""
        # Reconstruct trades with profits and returns
        trades_list = self.portfolio.trades
        if not trades_list:
            # No trades executed; return an empty DataFrame
            return pd.DataFrame()

        trades_df = pd.DataFrame(trades_list)
        all_trades = []

        for symbol in self.portfolio.symbols:
            symbol_trades = trades_df[trades_df['symbol'] == symbol].copy()
            # Process trades in pairs (entry and exit)
            i = 0
            while i < len(symbol_trades) - 1:
                entry_trade = symbol_trades.iloc[i]
                exit_trade = symbol_trades.iloc[i + 1]
                quantity = entry_trade['quantity']
                entry_price = entry_trade['price']
                exit_price = exit_trade['price']
                direction = entry_trade['direction']
                # Calculate profit
                if direction == 'BUY':
                    profit = (exit_price - entry_price) * quantity - entry_trade['commission'] - exit_trade['commission'] + exit_trade['roll']
                else:
                    profit = (entry_price - exit_price) * quantity - entry_trade['commission'] - exit_trade['commission'] + exit_trade['roll']
                trade = {
                    'symbol': symbol,
                    'entry_datetime': entry_trade['datetime'],
                    'exit_datetime': exit_trade['datetime'],
                    'direction': direction,
                    'quantity': quantity,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'profit': profit,
                    'roll' : exit_trade['roll'],
                    'return': profit / (entry_price * quantity) if entry_price * quantity != 0 else 0.0
                }
                all_trades.append(trade)
                i += 2  # Move to the next pair
        all_trades_df = pd.DataFrame(all_trades)
        return all_trades_df

    def construct_trade_list(self, symbol_trades):
        """Constructs a list of trades with entry and exit points."""
        trades = []
        position = 0
        entry_price = 0
        entry_quantity = 0
        entry_datetime = None

        for index, trade in symbol_trades.iterrows():
            direction = trade['direction']
            quantity = trade['quantity']
            price = trade['price']
            datetime = trade['datetime']
            commission = trade['commission']

            if direction == 'BUY':
                if position <= 0:
                    # Opening or increasing a long position
                    if position < 0:
                        # Closing short position first
                        exit_quantity = min(abs(position), quantity)
                        profit = (entry_price - price) * exit_quantity - commission
                        trade_return = profit / (abs(entry_price) * exit_quantity)
                        trades.append({
                            'symbol': trade['symbol'],
                            'entry_datetime': entry_datetime,
                            'exit_datetime': datetime,
                            'profit': profit,
                            'return': trade_return
                        })
                        position += exit_quantity
                        quantity -= exit_quantity
                    if quantity > 0:
                        # Opening new long position
                        position += quantity
                        entry_price = price
                        entry_quantity = quantity
                        entry_datetime = datetime
                else:
                    # Increasing existing long position
                    position += quantity
                    # Optionally update entry price
                    entry_price = (entry_price * entry_quantity + price * quantity) / (entry_quantity + quantity)
                    entry_quantity += quantity

            elif direction == 'SELL':
                if position >= 0:
                    # Closing or opening a short position
                    if position > 0:
                        # Closing long position first
                        exit_quantity = min(position, quantity)
                        profit = (price - entry_price) * exit_quantity - commission
                        trade_return = profit / (abs(entry_price) * exit_quantity)
                        trades.append({
                            'symbol': trade['symbol'],
                            'entry_datetime': entry_datetime,
                            'exit_datetime': datetime,
                            'profit': profit,
                            'return': trade_return
                        })
                        position -= exit_quantity
                        quantity -= exit_quantity
                    if quantity > 0:
                        # Opening new short position
                        position -= quantity
                        entry_price = price
                        entry_quantity = quantity
                        entry_datetime = datetime
                else:
                    # Increasing existing short position
                    position -= quantity
                    # Optionally update entry price
                    entry_price = (entry_price * abs(entry_quantity) + price * quantity) / (abs(entry_quantity) + quantity)
                    entry_quantity += quantity

        return trades

    def save_report(self, file_path, perfomance):
        """Saves the performance report to a file."""
        report_content =  [f"{key}: {value}\n" for key, value in perfomance.items()] #self.format_report()
        with open(file_path, 'w') as f:
            f.writelines(report_content)
        self.logger.info(f"The performance report was saved to a file: {file_path}")

    def generate_report(self, symbol):
        if self.trades.empty:
            return { 'WARNING': 'No trades available!' }       
        trades = self.trades[self.trades['symbol'] == symbol]
        if trades.empty:
            return { 'WARNING': 'No trades available!' }
    
        # Calculate metrics
        total_trades = len(trades)
        profits = trades['profit']
        gross_profit = profits[profits > 0].sum()
        gross_loss = profits[profits <= 0].sum()
        net_profit = gross_profit + gross_loss
        winning_trades = profits[profits > 0]
        losing_trades = profits[profits <= 0]
        win_percentage = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        avg_trade_profit = net_profit / total_trades if total_trades > 0 else 0
        max_win = winning_trades.max() if not winning_trades.empty else 0
        max_loss = losing_trades.min() if not losing_trades.empty else 0
        avg_win = winning_trades.mean() if not winning_trades.empty else 0
        avg_loss = losing_trades.mean() if not losing_trades.empty else 0
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else np.inf

        returns = trades['return']
        data = self.data_handler.symbol_data[symbol]
        equity = data[symbol]
        drawdown = (equity - equity.cummax()) * -1
        max_drawdown = drawdown.max()
        # Calculate drawdown duration
        duration = (drawdown != 0).astype(int).groupby((drawdown == 0).astype(int).cumsum()).cumsum()
        max_down_time = duration.max()
        max_drawdown_percentage = max_drawdown / equity.max() * 100 if equity.max() > 0 else 0.0
        avg_trade_duration = self.calculate_avg_trade_duration()
        max_trade_duration = self.calculate_max_trade_duration()
        time_in_market = self.calculate_time_in_market()
        max_open_trades = self.calculate_max_open_trades()
        max_loss_streak = self.calculate_max_loss_streak()

        if len(equity) > 0 and equity.iloc[0] != 0:
            annual_return = (equity.iloc[-1] / equity.iloc[0]) ** (252 / len(equity)) - 1
        else:
            annual_return = 0.0
        profit_factor = gross_profit / abs(gross_loss) if gross_loss != 0 else np.inf
        reward_risk_ratio = net_profit / abs(max_drawdown) if max_drawdown != 0 else np.inf
        r2_coefficient = self.calculate_r2_coefficient()
        sharpe_ratio = self.calculate_sharpe_ratio(returns)
        sortino_ratio = self.calculate_sortino_ratio(returns)
        annualized_stddev = returns.std() * np.sqrt(252)
        ulcer_index = self.calculate_ulcer_index()

        report = {
            'NEXUS v': __version__,
            'Strategy': self.portfolio.strategy_name,
            'Backtest duration': self.execution_time,
            'Bar period': self.get_bar_period(data),
            'Total bars' : len(data),
            'Start date' : self.data_handler.start_date,
            'End date' : self.data_handler.end_date,
            '':'',
            'Number of trades': total_trades,
            'Net profit': net_profit,
            'Gross profit': gross_profit,
            'Gross loss': gross_loss,
            'Winning %': win_percentage,
            'Avg trade profit': avg_trade_profit,
            'Max win': max_win,
            'Max loss': max_loss,
            'Win/Loss ratio': win_loss_ratio,

            'Max drawdown': max_drawdown,
            'Max drawdown %': max_drawdown_percentage,
            'Max down time' : max_down_time,
            'Avg trade, bars': avg_trade_duration,
            'Max trade, bars': max_trade_duration,
            'Time in market, %': time_in_market * 100,
            'Max open trades': max_open_trades,
            'Max loss streak': max_loss_streak,
            'Annual Return': annual_return * 100,
            'Profit Factor': profit_factor,
            'Reward Risk Ratio': reward_risk_ratio,
            'Sharpe Ratio': sharpe_ratio,
            'Sortino Ratio': sortino_ratio,
            'Annualized StdDev': annualized_stddev * 100,
            'R2 Coefficient': r2_coefficient,
            'Ulcer Index': ulcer_index * 100
        }
        return report

    # Helper methods to calculate the metrics
    def calculate_sharpe_ratio(self, returns, risk_free_rate=0.0):
        excess_returns = returns - risk_free_rate / 252
        std_dev = excess_returns.std()
        if std_dev == 0 or pd.isna(std_dev):
            return 0.0
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / std_dev
        return sharpe_ratio
    
    def calculate_sortino_ratio(self, returns, risk_free_rate=0.0):
        negative_returns = returns[returns < 0]
        downside_std = negative_returns.std()
        excess_returns = returns.mean() - risk_free_rate / 252
        sortino_ratio = np.sqrt(252) * excess_returns / downside_std if downside_std != 0 else np.nan
        return sortino_ratio

    def calculate_avg_trade_duration(self):
        durations = (self.trades['exit_datetime'] - self.trades['entry_datetime']).dt.days
        avg_duration = durations.mean()
        return avg_duration

    def calculate_max_trade_duration(self):
        durations = (self.trades['exit_datetime'] - self.trades['entry_datetime']).dt.days
        max_duration = durations.max()
        return max_duration

    def calculate_time_in_market(self):
        return 1 # ToDo implemnt

    def calculate_max_open_trades(self):
        # For this simple strategy, max open trades is 1
        # Modify as needed for more complex strategies
        return 1

    def calculate_max_loss_streak(self):
        profits = self.trades['profit']
        loss_streaks = (profits <= 0).astype(int).groupby((profits > 0).cumsum()).sum()
        max_loss_streak = loss_streaks.max()
        return max_loss_streak

    def calculate_r2_coefficient(self):
        returns = self.equity['returns']
        cumulative_returns = (1 + returns).cumprod()
        x = np.arange(len(cumulative_returns))
        y = np.log(cumulative_returns)
        slope, intercept = np.polyfit(x, y, 1)
        y_pred = intercept + slope * x
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
        return r2

    def calculate_ulcer_index(self):
        equity = self.equity['total']
        drawdowns = equity / equity.cummax() - 1
        squared_drawdowns = drawdowns ** 2
        ulcer_index = np.sqrt(squared_drawdowns.mean())
        return ulcer_index
    
    def get_bar_period(self, data):
        """Estimates the bar period from the index."""
        dates = data.index
        if len(dates) >= 2:
            delta = dates[1] - dates[0]
            if delta.days >= 1:
                return f"{delta.days} day(s)"
            else:
                hours = delta.seconds // 3600
                if hours >= 1:
                    return f"{hours} hour(s)"
                else:
                    minutes = delta.seconds // 60
                    return f"{minutes} minute(s)"
        else:
            return "Unknown"
        
    def save_trades_per_symbol(self, log_dir):
        """
        Saves reconstructed trades per symbol into log files under the given directory.

         Each file is named <symbol>.trades.txt and contains the trades for that symbol.
        """
        trades_df = self.construct_all_trades()
        if trades_df.empty:
            self.logger.info("save_trades_per_symbol() No trades to save!")
            return

        for symbol in self.portfolio.symbols:
            symbol_trades = trades_df[trades_df['symbol'] == symbol]
            if symbol_trades.empty:
                continue  # No trades for this symbol
            # Select the desired columns
            symbol_trades = symbol_trades[[
                'entry_datetime', 'exit_datetime', 'direction', 'quantity',
                'entry_price', 'exit_price', 'profit'
            ]]
            # Save to file
            file_path = os.path.join(log_dir, f"{symbol}.trades.txt")
            symbol_trades.to_csv(file_path, sep='\t', index=False)
            self.logger.info(f"save_trades_per_symbol: {file_path}")

    def save_trades(self, log_dir):
        """
        Saves all the trades to a CSV file.
        """
        if self.trades.empty:
            self.logger.info("save_trades() No trades to save!")
            return
        
        # return colum left
        df = self.trades.rename(columns={
            'symbol': 'Asset',
            'entry_datetime': 'Open', 
            'exit_datetime': 'Close',
            'direction': 'Type',
            'entry_price': 'Entry',
            'exit_price': 'Exit',
            'roll' : 'Roll',
            'profit': 'Profit'
            }, inplace=False)
        df = df[['Type', 'Asset', 'Open', 'Close', 'Entry', 'Exit', 'Profit', 'Roll']]
        file_path = os.path.join(log_dir, 'trades_log.csv')
        df.to_csv(file_path, index=False, float_format='%.6f')
        self.logger.info(f"All the trades was saved to a csv file: {file_path}")
        