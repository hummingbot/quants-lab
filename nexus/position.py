from datetime import datetime, timedelta

def format_value(value):
    return f"{value:.6f}" if value is not None else "None"

class Position:
    """Represents an open position in a symbol."""

    def __init__(self, symbol, asset):
        self.symbol = symbol
        self.asset = asset
        self.quantity = 0
        self.entry_price = 0.0
        self.entry_datetime = None
        self.exit_price = 0.0
        self.exit_datetime = None
        self.unrealized_pnl = 0.0
        self.realized_pnl = 0.0
        self.roll = 0.0
        self.stop_loss = None

    def update_on_fill(self, fill_event):
        """
        Updates the position based on a fill event.

        Parameters:
        - fill_event (FillEvent): The fill event containing trade execution details.
        """
        fill_dir = 1 if fill_event.direction == 'BUY' else -1
        quantity = fill_event.quantity
        price = fill_event.price
        timestamp = fill_event.timestamp
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        commission = fill_event.commission

        # Update position quantity
        prev_quantity = self.quantity
        self.quantity += fill_dir * quantity

        # If entering a new position
        if prev_quantity == 0 and self.quantity != 0:
            self.entry_price = price
            self.entry_datetime = timestamp
        # If closing a position
        elif self.quantity == 0:
            self.exit_price = price
            self.exit_datetime = timestamp
            self.roll = self.calculate_roll_cost(prev_quantity)
            pnl = self.calculate_pnl(prev_quantity, price, commission, self.roll)
            self.realized_pnl += pnl
            self.stop_loss = None
            self.entry_price = 0.0
            self.entry_datetime = None
        else:
            # Partially closing or increasing position
            pass  # Handle cases as needed

        # Update unrealized P&L
        self.unrealized_pnl = self.calculate_unrealized_pnl(current_price=price)

        # Update stop_loss if provided
        if fill_event.stop_loss is not None:
            self.stop_loss = fill_event.stop_loss

    def calculate_pnl(self, prev_quantity,  price, commission, roll):
        """
        Calculates the realized profit or loss when the position is closed.

        Parameters:
        - prev_quantity (int): The position quantity before the fill.
        - price (float): Execution price.
        - commission (float): Commission paid.
        - roll (float): roll cost.
        Returns:
        - float: Realized P&L.
        """
        pnl = (price - self.entry_price) * prev_quantity
        pnl -= commission
        pnl += roll # roll cost  is already negative
        return pnl

    def calculate_unrealized_pnl(self, current_price):
        """
        Calculates the unrealized profit or loss for the open position.

        Parameters:
        - current_price (float): The current market price.

        Returns:
        - float: Unrealized P&L.
        """
        if self.quantity == 0:
            return 0.0
        pnl = (current_price - self.entry_price) * self.quantity
        return pnl  

    def calculate_roll_cost(self, prev_quantity):
            """
            Calculates the roll cost for the position when it is closed.
            roll_long/roll_short is added to the trading cost for any trade 
            that was longer open than 12 hours at any new day, 
            including Saturday and Sunday.
            """
            import datetime

            if self.entry_datetime is None or self.exit_datetime is None:
                return 0.0  # Position not properly initialized

            # Define the rollover time in UTC (5:00 PM ET is 10:00 PM UTC)
            ROLLOVER_TIME = datetime.time(hour=22, minute=0, second=0)
            
            # Determine the first rollover date
            if self.entry_datetime.time() >= ROLLOVER_TIME:
                first_rollover_date = self.entry_datetime.date() + timedelta(days=1)
            else:
                first_rollover_date = self.entry_datetime.date()

            # Determine the last rollover date
            if self.exit_datetime.time() <= ROLLOVER_TIME:
                last_rollover_date = self.exit_datetime.date() - timedelta(days=1)
            else:
                last_rollover_date = self.exit_datetime.date()

            # Calculate the total number of days between the first and last rollover dates
            rollover_days = (last_rollover_date - first_rollover_date).days + 1

            # special tratment
            SOME_TIME = datetime.time(hour=4, minute=0, second=0)
            if self.entry_datetime.time() == SOME_TIME:
                rollover_days += 1

            if self.exit_datetime.time() == SOME_TIME:
                rollover_days -= 1

            if rollover_days <= 0:
                return 0.0  # No roll cost for positions closed before first rollover

            # Determine the roll rate
            if prev_quantity > 0:
                roll_rate = self.asset.roll_long
            elif prev_quantity < 0:
                roll_rate = self.asset.roll_short
            else:
                roll_rate = 0.0

            # Determine the number of lots
            number_of_lots = abs(prev_quantity) / self.asset.lot_amount

            # Total roll cost
            total_roll_cost = rollover_days * roll_rate * number_of_lots * self.asset.pip_cost
            
            #print(rollover_days, total_roll_cost)
            return total_roll_cost

    
    def reset(self):
        """
        Resets the position after it is closed.
        """
        self.__init__(self.symbol)  # Re-initialize the position

    def __str__(self):
        return (
            f"Position({self.symbol}, "
            f"qty: {self.quantity}, "
            f"open: {self.entry_datetime}, "
            f"close: {self.exit_datetime}, "
            f"entry: {format_value(self.entry_price)}, "
            f"exit: {format_value(self.exit_price)}, "
            f"unrealized_pnl: {self.unrealized_pnl:.6f}, "
            f"realized_pnl: {self.realized_pnl:.6f}, "
            f"roll: {self.roll:.6f}, "
            f"stop_loss: {format_value(self.stop_loss)})"
        )
    
    def calculate_number_of_rollovers(self, entry_datetime, exit_datetime):
        """
        Calculate the number of rollovers for a forex trade.
        A rollover occurs for each day the trade was open for more than 12 hours.

        Parameters:
        - entry_datetime: datetime object representing when the trade was opened.
        - exit_datetime: datetime object representing when the trade was closed.

        Returns:
        - number_of_rollovers: int, the total number of rollovers.
        """
        number_of_rollovers = 0

        current_date = entry_datetime.date()
        end_date = exit_datetime.date()
        
        while current_date <= end_date:
            # Define the start and end of the current day
            start_of_day = datetime.combine(current_date, datetime.min.time())
            end_of_day = datetime.combine(current_date, datetime.max.time())

            # Calculate the trade's active time on the current day
            trade_start = max(entry_datetime, start_of_day)
            trade_end = min(exit_datetime, end_of_day)
            open_duration = trade_end - trade_start

            # Check if the trade was open for more than 12 hours on this day
            if open_duration.total_seconds() > 12 * 3600:
                number_of_rollovers += 1

            # Move to the next day
            current_date += timedelta(days=1)

        return number_of_rollovers
    
    #from datetime import datetime, timedelta, time

    def calculate_number_of_rollovers2(self, entry_datetime, exit_datetime):
        """
        Calculate the number of rollover days for a forex position held between entry_datetime and exit_datetime.
        Assumes that both entry_datetime and exit_datetime are in UTC.

        Parameters:
        - entry_datetime (datetime): The datetime when the position was opened (UTC).
        - exit_datetime (datetime): The datetime when the position was closed (UTC).

        Returns:
        - int: The total number of rollover days.
        """

        # Ensure that entry_datetime and exit_datetime are timezone-aware UTC datetimes
        #if entry_datetime.tzinfo is None or entry_datetime.utcoffset() != timedelta(0):
        #    raise ValueError("entry_datetime must be a timezone-aware UTC datetime.")
        #if exit_datetime.tzinfo is None or exit_datetime.utcoffset() != timedelta(0):
        #    raise ValueError("exit_datetime must be a timezone-aware UTC datetime.")

        # Define the rollover time in UTC (5:00 PM ET is 10:00 PM UTC)
        rollover_time_utc = datetime.time(hour=22, minute=0, second=0)  # 10:00 PM UTC

        # Determine the first rollover date
        if entry_datetime.time() >= rollover_time_utc:
            first_rollover_date = entry_datetime.date() + timedelta(days=1)
        else:
            first_rollover_date = entry_datetime.date()

        # Determine the last rollover date
        if exit_datetime.time() <= rollover_time_utc:
            last_rollover_date = exit_datetime.date() - timedelta(days=1)
        else:
            last_rollover_date = exit_datetime.date()

        # Calculate the total number of days between the first and last rollover dates
        total_days = (last_rollover_date - first_rollover_date).days + 1

        rollover_days = 0
        for i in range(total_days):
            rollover_date = first_rollover_date + timedelta(days=i)
            weekday = rollover_date.weekday()

            # Triple rollover on Wednesdays (weekday == 2)
            if weekday == 2:
                rollover_days += 3
            else:
                rollover_days += 1

        return rollover_days
