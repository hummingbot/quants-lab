# nexus/asset.py

import csv

class Asset:
    """
    Represents a trading asset with all necessary parameters.
    """

    def __init__(self, name, price, spread, roll_long, roll_short, pip, pip_cost, margin_cost,
                 market, lot_amount, commission, symbol, asset_type):
        self.name = name
        self.price = float(price)
        self.spread = float(spread)
        self.roll_long = float(roll_long)
        self.roll_short = float(roll_short)
        self.pip = float(pip)
        self.pip_cost = float(pip_cost)
        self.margin_cost = float(margin_cost)
        self.market = market
        self.lot_amount = float(lot_amount)
        self.commission = float(commission)
        self.symbol = symbol
        self.asset_type = asset_type.lower()  # Ensure consistency

        # Add a check to ensure spread is in price units
        # If spread is in pips, convert it to price units
        #if self.asset_type == 'forex':
        #    self.spread_in_pips = self.spread
        #    self.spread = self.spread * self.pip  # Convert spread to price units

    @classmethod
    def from_csv_row(cls, row):
        return cls(
            name=row['Name'],
            price=row['Price'],
            spread=row['Spread'],
            roll_long=row['RollLong'],
            roll_short=row['RollShort'],
            pip=row['PIP'],
            pip_cost=row['PIPCost'],
            margin_cost=row['MarginCost'],
            market=row['Market'],
            lot_amount=row['LotAmount'],
            commission=row['Commission'],
            symbol=row['Symbol'],
            asset_type=row.get('AssetType', 'forex')  # Default to 'forex' if not specified
        )

def load_assets(csv_file_path):
    """
    Loads assets from a CSV file.

    Parameters:
    - csv_file_path (str): Path to the assets CSV file.

    Returns:
    - dict: A dictionary of Asset objects keyed by their symbol.
    """
    assets = {}
    with open(csv_file_path, mode='r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            asset = Asset.from_csv_row(row)
            assets[asset.symbol] = asset
    return assets
