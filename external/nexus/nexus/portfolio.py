import logging
from nexus.abc import Portfolio
from nexus.event import OrderEvent
from nexus.position import Position

class NaivePortfolio(Portfolio):
    """A simple portfolio that tracks positions and P&L."""

    def __init__(self, events, data_handler, symbols, initial_capital,
                 strategy_name='strategy', assets=None):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.debug("__init__ NaivePortfolio")
        self.events = events
        self.data_handler = data_handler
        self.symbols = symbols
        self.initial_capital = initial_capital
        self.strategy_name = strategy_name
        if assets is None:
            from nexus.asset import Asset
            assets = {s: Asset(
                name=s,
                price=0.0,
                spread=0.0,
                roll_long=0.0,
                roll_short=0.0,
                pip=1.0,
                pip_cost=1.0,
                margin_cost=0.0,
                market='',
                lot_amount=1.0,
                commission=0.0,
                symbol=s,
                asset_type='forex'
            ) for s in symbols}
        self.assets = assets
        self.all_positions = self.construct_all_positions()
        self.current_positions = {symbol: Position(symbol, assets[symbol]) for symbol in self.symbols}
        self.all_holdings = self.construct_all_holdings()
        self.current_holdings = self.construct_current_holdings()
        self.trades = []  # List to store trades
        if self.data_handler is not None:
            for symbol in self.symbols:
                data = self.data_handler.symbol_data[symbol]
                data[symbol] = 0.0  # Column to keep equity/PnL per bar

    def get_symbol_data(self, symbol):
        return self.data_handler.symbol_data[symbol]

    def construct_all_positions(self):
        d = {symbol: 0 for symbol in self.symbols}
        d['datetime'] = None
        return [d]

    def construct_all_holdings(self):
        d = {symbol: 0.0 for symbol in self.symbols}
        d['datetime'] = None
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return [d]

    def construct_current_holdings(self):
        d = {symbol: 0.0 for symbol in self.symbols}
        d['cash'] = self.initial_capital
        d['commission'] = 0.0
        d['total'] = self.initial_capital
        return d

    def update_time_index(self, event):
        latest_datetime = event.data.name

        # Update positions
        positions = {symbol: self.current_positions[symbol].quantity for symbol in self.symbols}
        positions['datetime'] = latest_datetime
        self.all_positions.append(positions)

        # Update holdings
        #print('Updating positions')
        holdings = {symbol: 0.0 for symbol in self.symbols}
        holdings['datetime'] = latest_datetime
        holdings['cash'] = self.current_holdings['cash']
        holdings['commission'] = self.current_holdings['commission']
        holdings['total'] = self.current_holdings['cash']
        for symbol in self.symbols:
            # Get the latest price for the symbol
            price = event.data['Close'] if event.symbol == symbol else self.get_latest_price(symbol)
            market_value = self.current_positions[symbol].quantity * price
            holdings[symbol] = market_value
            holdings['total'] += market_value
            #
            position = self.current_positions[symbol]
            equty = position.realized_pnl + position.calculate_unrealized_pnl(price)
            self.data_handler.symbol_data[symbol].at[latest_datetime, symbol] = equty
        self.all_holdings.append(holdings)

    def get_latest_price(self, symbol):
        """Retrieves the latest price for a symbol from the data handler."""
        price = self.data_handler.get_latest_bar_value(symbol, 'Close')
        return price if price is not None else 0.0

    def generate_naive_order(self, signal):
        order_type = 'MARKET'
        price = None
        quantity = signal.quantity  # Use the quantity from the signal
        if signal.signal_type == 'LONG':
            direction = 'BUY'
        elif signal.signal_type == 'SHORT':
            direction = 'SELL'
        elif signal.signal_type == 'EXIT':
            # Determine current position to close
            if self.current_positions[signal.symbol].quantity > 0:
                direction = 'SELL'
            elif self.current_positions[signal.symbol].quantity < 0:
                direction = 'BUY'
            else:
                # No position to exit
                return None
        else:
            return None
        # Convert to a limit order if a limit price is specified
        if getattr(signal, 'limit_price', None) is not None:
            order_type = 'LIMIT'
            price = signal.limit_price

        return OrderEvent(
            symbol=signal.symbol,
            timestamp=signal.timestamp,
            order_type=order_type,
            quantity=quantity,
            direction=direction,
            price=price,
            stop_loss=getattr(signal, 'stop_loss', None)
        )

    def update_signal(self, event):
        order_event = self.generate_naive_order(event)
        if order_event:
            self.events.put(order_event)

    def update_fill(self, event):
        """
        Updates the portfolio based on a FillEvent.
        """
        symbol = event.symbol
        position = self.current_positions[symbol]

        # Update the position with the fill event
        position.update_on_fill(event)

        # Update holdings
        fill_dir = 1 if event.direction == 'BUY' else -1
        quantity = event.quantity
        price = event.price
        commission = event.commission

        cost = fill_dir * price * quantity + commission
        self.current_holdings['cash'] -= cost
        self.current_holdings['commission'] += commission
        self.current_holdings[symbol] = position.quantity * price
        
        # Update total holdings
        self.current_holdings['total'] = self.current_holdings['cash']
        for sym in self.symbols:
            self.current_holdings['total'] += self.current_holdings[sym]

        # Record the trade
        trade = {
            'symbol': symbol,
            'datetime': event.timestamp,
            'direction': event.direction,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'roll' : position.roll
            #'type': 'STOP_LOSS'
        }
        self.trades.append(trade)

    def update_market(self, event):
        """
        Checks if any positions have hit their stop-loss levels and updates the portfolio accordingly.
        """
        for symbol in self.symbols:
            position = self.current_positions[symbol]
            if position.quantity != 0 and position.stop_loss is not None:
                # Get the latest price for the symbol
                price = self.get_latest_price(symbol)
                # Check for stop loss trigger
                if position.quantity > 0 and price <= position.stop_loss:
                    # Long position stop loss hit
                    self.close_position(symbol, price, event.data.name)
                elif position.quantity < 0 and price >= position.stop_loss:
                    # Short position stop loss hit
                    self.close_position(symbol, price, event.data.name)

    def close_position(self, symbol, price, datetime):
        """
        Closes the position at the specified price and time.
        """
        position = self.current_positions[symbol]
        quantity = abs(position.quantity)
        direction = 'SELL' if position.quantity > 0 else 'BUY'
        fill_dir = -1 if direction == 'SELL' else 1
        commission = self.calculate_commission(quantity)

        # Update positions
        position.quantity = 0
        position.stop_loss = None

        # Update holdings
        cost = fill_dir * price * quantity + commission
        self.current_holdings['cash'] -= cost
        self.current_holdings['commission'] += commission
        self.current_holdings[symbol] = 0.0  # No position now

        # Update total holdings
        self.current_holdings['total'] = self.current_holdings['cash']
        for sym in self.symbols:
            self.current_holdings['total'] += self.current_holdings[sym]

        # Record the trade
        trade = {
            'symbol': symbol,
            'datetime': datetime,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'commission': commission,
            'type': 'STOP_LOSS'
        }
        self.trades.append(trade)
        self.logger.info(f"Stop loss hit for {symbol} at {price}, position closed.")

    def calculate_commission(self, quantity):
        # Simple commission model
        return max(1.0, 0.0001 * quantity)