import asyncio
import os
import pandas_ta as ta  # noqa: F401
import json
import time
from datetime import datetime

import joblib
import paho.mqtt.client as mqtt
from hummingbot.data_feed.candles_feed.candles_factory import CandlesFactory
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig


class PredictionService:
    def __init__(self, scaler_path: str, model_path: str, candles_config: CandlesConfig, mqtt_broker="localhost", mqtt_port=1883):
        self.scaler = joblib.load(scaler_path)
        self.model = joblib.load(model_path)
        self.candles_config = candles_config
        self.candles = CandlesFactory.get_candle(candles_config=self.candles_config)
        self.candles.start()
        
        # Set up MQTT client
        self.mqtt_client = mqtt.Client()
        self.mqtt_broker = mqtt_broker
        self.mqtt_port = mqtt_port
        self.setup_mqtt()
        
    def setup_mqtt(self):
        """Configure and connect to the MQTT broker"""
        # Setup connection callbacks
        self.mqtt_client.on_connect = self.on_connect
        self.mqtt_client.on_disconnect = self.on_disconnect
        
        # Connect to the broker
        try:
            self.mqtt_client.connect(self.mqtt_broker, self.mqtt_port)
            self.mqtt_client.loop_start()  # Start the loop in a separate thread
            print(f"Connected to MQTT broker at {self.mqtt_broker}:{self.mqtt_port}")
        except Exception as e:
            print(f"Failed to connect to MQTT broker: {e}")
    
    def on_connect(self, client, userdata, flags, rc):
        """Callback for when the client receives a CONNACK response from the server"""
        if rc == 0:
            print("Successfully connected to MQTT broker")
        else:
            print(f"Failed to connect to MQTT broker with code {rc}")
    
    def on_disconnect(self, client, userdata, rc):
        """Callback for when the client disconnects from the server"""
        if rc != 0:
            print(f"Unexpected disconnection from MQTT broker: {rc}")
            # Try to reconnect
            print("Attempting to reconnect...")
            self.mqtt_client.loop_stop()
            self.setup_mqtt()

    def publish_prediction(self, prediction, trading_pair):
        """Publish prediction to MQTT broker"""
        # Format the prediction according to the specified format
        signal = {
            "id": int(time.time() * 1000),  # current timestamp in milliseconds
            "trading_pair": trading_pair,
            "probabilities": prediction.tolist()  # Convert numpy array to list
        }
        
        # Convert to JSON and publish
        try:
            message = json.dumps(signal)
            self.mqtt_client.publish('hbot/9d0a0eeb06739f1ee4214d3632841603908b55fe/ML_SIGNALS', message)
            print(f"Published prediction: {message}")
        except Exception as e:
            print(f"Failed to publish prediction: {e}")

    async def prediction_loop(self):
        while True:
            # Get new data
            if self.candles.ready:
                candles_df = self.candles.candles_df.copy()
                # Bollinger Bands with different lengths
                candles_df["target"] = candles_df["close"].rolling(200).std() / candles_df["close"]
                candles_df.ta.bbands(length=20, std=2, append=True)  # Standard BB
                candles_df.ta.bbands(length=50, std=2, append=True)  # Longer term BB

                # MACD with different parameters
                candles_df.ta.macd(fast=12, slow=26, signal=9, append=True)  # Standard MACD
                candles_df.ta.macd(fast=8, slow=21, signal=5, append=True)  # Faster MACD

                # RSI with different lengths
                candles_df.ta.rsi(length=14, append=True)  # Standard RSI
                candles_df.ta.rsi(length=21, append=True)  # Longer RSI

                # Moving averages
                candles_df.ta.sma(length=20, append=True)  # Short MA
                candles_df.ta.sma(length=50, append=True)  # Medium MA
                candles_df.ta.ema(length=20, append=True)  # Short EMA
                candles_df.ta.ema(length=50, append=True)  # Medium EMA

                # Volatility and momentum indicators
                candles_df.ta.atr(length=14, append=True)  # ATR
                candles_df.ta.stoch(k=14, d=3, append=True)  # Stochastic
                candles_df.ta.adx(length=14, append=True)  # ADX
                
                # Save timestamp before dropping it
                latest_timestamp = candles_df['timestamp'].iloc[-1] if 'timestamp' in candles_df.columns else int(time.time() * 1000)
                
                columns_to_drop = ['timestamp', 'taker_buy_base_volume', 'volume']
                candles_df = candles_df.drop(columns=columns_to_drop)
                # 2. Convert prices to returns
                price_columns = ['open', 'high', 'low', 'close']
                for col in price_columns:
                    candles_df[f'{col}_ret'] = candles_df[col].pct_change()
                candles_df = candles_df.drop(columns=price_columns)

                # 3. Create buy/sell volume ratio
                candles_df['buy_volume_ratio'] = candles_df['taker_buy_quote_volume'] / candles_df[
                    'quote_asset_volume']
                candles_df = candles_df.drop(columns=['taker_buy_quote_volume'])

                # 4. Drop any rows with NaN values (first row will have NaN due to returns calculation)
                candles_df = candles_df.dropna()

                # 5. Get all numeric columns for scaling (excluding the target 'close_type')
                numeric_columns = candles_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

                # 6. Apply StandardScaler to all numeric columns
                candles_df[numeric_columns] = self.scaler.transform(candles_df[numeric_columns])
                prediction = self.model.predict_proba(candles_df)[-1]
                
                # Publish the prediction to MQTT
                self.publish_prediction(prediction, self.candles_config.trading_pair)

            await asyncio.sleep(0.05)
            
    def cleanup(self):
        """Clean up resources when done"""
        if hasattr(self, 'mqtt_client'):
            self.mqtt_client.loop_stop()
            self.mqtt_client.disconnect()
            print("Disconnected from MQTT broker")


async def main():
    root_path = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    scaler_path = root_path + "/models/scaler.pkl"
    model_path = root_path + "/models/model.joblib"
    candles_config = CandlesConfig(connector="binance", trading_pair="BTC-USDT", interval="1s", max_records=1000)
    
    # Create the prediction service with local MQTT broker configuration
    prediction_service = PredictionService(
        scaler_path=scaler_path,
        model_path=model_path,
        candles_config=candles_config,
        mqtt_broker="localhost",
        mqtt_port=1883  # Default MQTT port, matching your EMQX broker
    )
    
    try:
        await prediction_service.prediction_loop()
    except KeyboardInterrupt:
        print("Shutting down prediction service...")
    finally:
        prediction_service.cleanup()

if __name__ == "__main__":
    asyncio.run(main())