import asyncio
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Any, Dict

import pandas as pd
import numpy as np
from dotenv import load_dotenv

from core.data_sources import CLOBDataSource
from core.services.timescale_client import TimescaleClient
from core.task_base import BaseTask
from core.backtesting.triple_barrier_method import triple_barrier_method
from sklearn.preprocessing import StandardScaler
import pandas_ta as ta
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json


logging.basicConfig(level=logging.INFO)
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
load_dotenv()


class Bitcoinenaitor(BaseTask):
    def __init__(self, name: str, frequency: timedelta, config: Dict[str, Any]):
        super().__init__(name, frequency, config)
        self.connector_name = config["connector_name"]
        self.days_for_training = config.get("days_for_training", 7)
        self.interval = config.get("interval", "1s")
        self.trading_pair = config.get("trading_pair", "BTC-USDT")
        self.target_multiplier = config.get("target_multiplier", 3.0)
        self.tl = config.get("tl", 300)
        self.std_span = config.get("std_span", 200)
        self.trade_cost = config.get("trade_cost", 0.00)
        
        # MLflow settings
        self.mlflow_tracking_uri = config.get("mlflow_config", {}).get("tracking_uri", "http://localhost:5000")
        self.experiment_name = config.get("mlflow_config", {}).get("experiment_name", f"{self.connector_name}_{self.trading_pair}_{self.interval}")
        self.n_estimators = config.get("model_params", {}).get("n_estimators", 500)
        self.max_depth = config.get("model_params", {}).get("max_depth", 3)
        self.cv_folds = config.get("model_params", {}).get("cv_folds", 5)

        self.timescale_client = TimescaleClient(
            host=self.config["timescale_config"].get("db_host", "localhost"),
            port=self.config["timescale_config"].get("db_port", 5432),
            user=self.config["timescale_config"].get("db_user", "admin"),
            password=self.config["timescale_config"].get("db_password", "admin"),
            database=self.config["timescale_config"].get("db_name", "timescaledb")
        )

    async def execute(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f UTC")
        logging.info(f"{now} - Starting ml pipeline for {self.connector_name} {self.trading_pair}")
        end_time = datetime.now(timezone.utc)
        start_time = pd.Timestamp(time.time() - self.days_for_training * 24 * 60 * 60,unit="s").tz_localize(timezone.utc).timestamp()
        logging.info(f"{now} - Start date: {start_time}, End date: {end_time}")
        await self.timescale_client.connect()
        candles = await self.timescale_client.get_candles_last_days(self.connector_name, self.trading_pair, self.interval, self.days_for_training)
        df = candles.data
        df["side"] = 1
        logging.info("Starting meta labeling")
        df_with_tbm = triple_barrier_method(df, tp=self.target_multiplier, sl=self.target_multiplier, tl=self.tl, std_span=self.std_span, trade_cost=self.trade_cost)
        logging.info(f"Meta labeling complete, adding features ")
        df_processed, scaler = self.add_features(df_with_tbm)
        
        # Train and evaluate model with MLflow tracking
        logging.info("Training and evaluating model with MLflow tracking")
        self.train_and_evaluate_model(df_processed, scaler)
        
        await self.timescale_client.close()

    def train_and_evaluate_model(self, df_processed, scaler):
        # Configure MLflow
        mlflow.set_tracking_uri(self.mlflow_tracking_uri)
        
        # Create or get the experiment
        experiment = mlflow.get_experiment_by_name(self.experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(self.experiment_name)
        else:
            experiment_id = experiment.experiment_id
        
        # Start an MLflow run
        with mlflow.start_run(experiment_id=experiment_id, run_name=f"{self.connector_name}_{self.trading_pair}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_params({
                "connector_name": self.connector_name,
                "trading_pair": self.trading_pair,
                "interval": self.interval,
                "days_for_training": self.days_for_training,
                "target_multiplier": self.target_multiplier,
                "tl": self.tl,
                "std_span": self.std_span,
                "trade_cost": self.trade_cost,
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "cv_folds": self.cv_folds
            })
            
            # Prepare features and target
            feature_columns = [col for col in df_processed.columns if col != 'close_type']
            X = df_processed[feature_columns]
            y = df_processed['close_type']
            
            # Log initial class distribution
            class_dist = y.value_counts().sort_index().to_dict()
            mlflow.log_params({f"class_{k}_count": v for k, v in class_dist.items()})
            logging.info(f"Initial class distribution: {class_dist}")
            
            # Balance the dataset
            # Get the size of the smaller classes
            target_size = df_processed[df_processed['close_type'] != 0].shape[0] // 2
            df_neg = df_processed[df_processed['close_type'] == -1]
            df_pos = df_processed[df_processed['close_type'] == 1]
            df_mid = df_processed[df_processed['close_type'] == 0].sample(n=target_size, random_state=42)
            
            # Combine the balanced dataset
            balanced_df = pd.concat([df_neg, df_mid, df_pos])
            
            X_balanced = balanced_df[feature_columns]
            y_balanced = balanced_df['close_type']
            
            # Log balanced class distribution
            balanced_class_dist = y_balanced.value_counts().sort_index().to_dict()
            mlflow.log_params({f"balanced_class_{k}_count": v for k, v in balanced_class_dist.items()})
            logging.info(f"Balanced class distribution: {balanced_class_dist}")
            
            # Initialize the model
            model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced',
            )
            
            # Perform cross-validation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            cv_results = cross_validate(
                model, 
                X_balanced, 
                y_balanced, 
                cv=cv,
                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'],
                return_train_score=True,
                return_estimator=True
            )
            
            # Log cross-validation metrics
            for metric in ['test_accuracy', 'test_precision_macro', 'test_recall_macro', 'test_f1_macro']:
                mean_metric = cv_results[metric].mean()
                std_metric = cv_results[metric].std()
                mlflow.log_metric(f"cv_mean_{metric}", mean_metric)
                mlflow.log_metric(f"cv_std_{metric}", std_metric)
                logging.info(f"{metric}: {mean_metric:.4f} ± {std_metric:.4f}")
            
            # Train final model on the full balanced dataset
            logging.info("Training final model on full balanced dataset")
            model.fit(X_balanced, y_balanced)
            
            # Make a train/test split for final evaluation
            X_train, X_test, y_train, y_test = train_test_split(
                X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced
            )
            
            # Evaluate on test set
            y_pred = model.predict(X_test)
            
            # Calculate and log metrics
            accuracy = accuracy_score(y_test, y_pred)
            precision = precision_score(y_test, y_pred, average='macro')
            recall = recall_score(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            
            mlflow.log_metrics({
                "final_accuracy": accuracy,
                "final_precision": precision,
                "final_recall": recall,
                "final_f1": f1
            })
            
            # Log confusion matrix as a figure
            cm = confusion_matrix(y_test, y_pred)
            cm_dict = {
                "matrix": cm.tolist(),
                "labels": sorted(list(set(y_test)))
            }
            with open("confusion_matrix.json", "w") as f:
                json.dump(cm_dict, f)
            mlflow.log_artifact("confusion_matrix.json")
            
            # Log classification report
            report = classification_report(y_test, y_pred, output_dict=True)
            with open("classification_report.json", "w") as f:
                json.dump(report, f)
            mlflow.log_artifact("classification_report.json")
            
            # Log feature importances
            feature_importance = {feature: importance for feature, importance in 
                                 zip(feature_columns, model.feature_importances_)}
            sorted_feature_importance = dict(sorted(feature_importance.items(), 
                                                    key=lambda item: item[1], reverse=True))
            with open("feature_importance.json", "w") as f:
                json.dump(sorted_feature_importance, f)
            mlflow.log_artifact("feature_importance.json")
            
            # Log model and scaler
            mlflow.sklearn.log_model(model, "model")
            mlflow.sklearn.log_model(scaler, "scaler")
            
            # Create model signature with input and output schema for future inference
            model_info = mlflow.sklearn.log_model(model, "model")
            
            logging.info(f"Model and artifacts logged in MLflow run: {mlflow.active_run().info.run_id}")
            logging.info(f"MLflow UI URL: {self.mlflow_tracking_uri}/#/experiments/{experiment_id}")
            
            return model, scaler

    @staticmethod
    def add_features(df):
        # Create a copy to work with
        df_with_indicators = df.copy()

        # Bollinger Bands with different lengths
        df_with_indicators.ta.bbands(length=20, std=2, append=True)  # Standard BB
        df_with_indicators.ta.bbands(length=50, std=2, append=True)  # Longer term BB

        # MACD with different parameters
        df_with_indicators.ta.macd(fast=12, slow=26, signal=9, append=True)  # Standard MACD
        df_with_indicators.ta.macd(fast=8, slow=21, signal=5, append=True)  # Faster MACD

        # RSI with different lengths
        df_with_indicators.ta.rsi(length=14, append=True)  # Standard RSI
        df_with_indicators.ta.rsi(length=21, append=True)  # Longer RSI

        # Moving averages
        df_with_indicators.ta.sma(length=20, append=True)  # Short MA
        df_with_indicators.ta.sma(length=50, append=True)  # Medium MA
        df_with_indicators.ta.ema(length=20, append=True)  # Short EMA
        df_with_indicators.ta.ema(length=50, append=True)  # Medium EMA

        # Volatility and momentum indicators
        df_with_indicators.ta.atr(length=14, append=True)  # ATR
        df_with_indicators.ta.stoch(k=14, d=3, append=True)  # Stochastic
        df_with_indicators.ta.adx(length=14, append=True)  # ADX

        # Replace df_with_tbm with df_with_indicators for further processing
        df_processed = df_with_indicators.copy()

        # df_processed.reset_index(inplace=True, drop=True)

        # 1. Remove unnecessary columns
        columns_to_drop = ['timestamp', 'taker_buy_base_volume', 'volume',
                           'close_time', 'real_class', 'ret', 'tp', 'sl', 'take_profit_time', 'stop_loss_time', 'tl',
                           'side']
        df_processed = df_processed.drop(columns=columns_to_drop)
        # 2. Convert prices to returns
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            df_processed[f'{col}_ret'] = df_processed[col].pct_change()
        df_processed = df_processed.drop(columns=price_columns)

        # 3. Create buy/sell volume ratio
        df_processed['buy_volume_ratio'] = df_processed['taker_buy_quote_volume'] / df_processed['quote_asset_volume']
        df_processed = df_processed.drop(columns=['taker_buy_quote_volume'])

        # 4. Drop any rows with NaN values (first row will have NaN due to returns calculation)
        df_processed = df_processed.dropna()

        # 5. Get all numeric columns for scaling (excluding the target 'close_type')
        numeric_columns = df_processed.select_dtypes(include=['float64', 'int64']).columns.tolist()
        numeric_columns.remove('close_type')  # Don't scale the target variable

        # 6. Apply StandardScaler to all numeric columns
        scaler = StandardScaler()
        df_processed[numeric_columns] = scaler.fit_transform(df_processed[numeric_columns])
        return df_processed, scaler

async def main(config):
    ml_pipeline = Bitcoinenaitor(
        name="Candles Downloader",
        frequency=timedelta(hours=1),
        config=config
    )
    await ml_pipeline.execute()

if __name__ == "__main__":
    timescale_config = {
        "host": os.getenv("TIMESCALE_HOST", "localhost"),
        "port": os.getenv("TIMESCALE_PORT", 5432),
        "user": os.getenv("TIMESCALE_USER", "admin"),
        "password": os.getenv("TIMESCALE_PASSWORD", "admin"),
        "database": os.getenv("TIMESCALE_DB", "timescaledb")
    }
    
    mlflow_config = {
        "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
        "experiment_name": os.getenv("MLFLOW_EXPERIMENT_NAME", "bitcoinenaitor")
    }
    
    model_params = {
        "n_estimators": 500,
        "max_depth": 3,
        "cv_folds": 5
    }
    
    config = {
        "connector_name": "binance",
        "interval": "1s",
        "days_for_training": 3,
        "trading_pair": "BTC-USDT",
        "timescale_config": timescale_config,
        "mlflow_config": mlflow_config,
        "model_params": model_params
    }
    asyncio.run(main(config))
