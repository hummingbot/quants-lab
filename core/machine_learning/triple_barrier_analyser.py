import pickle
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from scipy import stats
from scipy.stats import chi2_contingency

from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
)

from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler

from core.backtesting.triple_barrier_method import triple_barrier_method
from core.machine_learning.features import Features
from core.machine_learning.model_config import ModelConfig


def get_random_matrix(rows=None):
    # Specify the dimensions of the matrix and the range of values
    if rows is not None:
        pass
    else:
        rows = random.randint(1, 4)

    rows = rows
    cols = 4
    min_value = 1
    max_value = 10

    # Generate a random matrix with restricted values
    random_matrix = [[random.randint(min_value, max_value) for _ in range(cols)] for _ in range(rows)]
    return random_matrix


class TripleBarrierAnalyser:
    """
    Triple Barrier Analyser by Mr. Ghetman @blaspalmisciano
    """
    def __init__(self,
                 df: pd.DataFrame = pd.DataFrame(),
                 features_dict: Dict[str, Any] = None,
                 external_feat: Dict[str, Any] = None,
                 tp: float = 0.001,
                 sl: float = 0.001,
                 tl: int = 3600,
                 trade_cost: float = 0.0006,
                 dump_pickle: bool = True,
                 ):
        self.df = df
        self.features_dict = features_dict
        self.external_feat = external_feat
        self.tp = tp
        self.sl = sl
        self.tl = tl
        self.trade_cost = trade_cost
        self.dump_pickle = dump_pickle
        self.trgt = None
        self.label_encoder = None
        self.transformer = None
        self.classes = None
        self.X = None
        self.y = None,
        self.X_train = None,
        self.X_test = None,
        self.y_train = None,
        self.y_test = None,
        self.y_pred = None,
        self.y_pred_train = None,
        self.best_random = None
        self.X_columns = [],
        self.visualize = False,
        self.to_log = None,
        self.categorical = None,
        self.to_scale = None,
        self.results = pd.DataFrame()

        self.df_null_values = None

    def prepare_data(self, candles_df: pd.DataFrame):
        df = triple_barrier_method(candles_df, tp=self.tp, sl=self.sl, tl=self.tl, trade_cost=self.trade_cost)
        if self.external_feat:
            df = self.add_features(df, self.external_feat)
        return df

    @staticmethod
    def add_features(candles_df: pd.DataFrame, external_feat: Dict[str, Any] = None):
        ft = Features(external_feat, candles_df)
        return ft.add_features()

    def transform_train(self, features_df: pd.DataFrame, model_config: ModelConfig):
        self.model_pre_processing(features_df=features_df)
        self.classify_model(model_config)

        if self.dump_pickle:
            data_to_pickle = {
                'pipeline': Pipeline([
                    ('transformer', self.transformer),
                    ('prediction', self.best_random)
                ]),
                'extra_features': self.external_feat
            }
            with open('pipeline.pkl', 'wb') as f:
                pickle.dump(data_to_pickle, f)

    def model_pre_processing(self,
                             features_df: pd.DataFrame,
                             admitted_cols: List[str] = None,
                             test_size: float = 0.2):
        df = features_df.copy()
        X, y = self.column_transform(features_df=df)
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        y = label_encoder.transform(y)
        self.classes = label_encoder.classes_
        self.label_encoder = label_encoder
        self.X = X
        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, test_size=test_size)

    def column_transform(self, features_df: pd.DataFrame, training=True):
        # scale_pipe = make_pipeline(StandardScaler())
        # log_pipe = make_pipeline(PowerTransformer())
        # categorical_pipe = make_pipeline(OneHotEncoder(sparse=False, handle_unknown="ignore"))

        # self.to_log = ['volume', 'MACD_12_26_9_ALPHA', 'RSI_14_MEMO']
        # self.categorical = ['MACD_12_26_9_SIDE', 'RSI_14_SIDE', 'MACD_12_26_9_CROSS']
        # self.to_scale = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACD_12_26_9_VALUE', 'RSI_14']
        # cols = self.categorical + self.to_log + self.to_scale
        df = features_df.copy()
        cols = features_df.select_dtypes(include=['int', 'float', 'bool'])
        cols = [
            column for column in cols if column not in [
                'ret_2','tp','sl', 't1', 'Pt1', 'ret', 'stop_loss_time', 'take_profit_time', 'close_price','index','tl',
                'ret/trgt', 'time','close_time','real_class','marker','index','Unnamed: 0', 'ignore',"datetime",
                'trgt','open','volume','close','taker_buy_base_volume','qav','low','high','taker_buy_quote_volume','num_trades',
                'trade_pnl', 'net_pnl','real_class','profitable','signal', 'side','take_profit_price',
                'stop_loss_price','timestamp'
            ]
        ]
        transformer = ColumnTransformer(
            transformers=[
                # ("subset", lambda x: x[cols], self.df),
                # ("scale", scale_pipe, self.to_scale),
                # ("log_transform", log_pipe, self.to_log),
                # ("oh_encode", categorical_pipe, self.categorical)
            ],
            remainder='passthrough'
        )
        subset = [c for c in df.columns if c not in ['taker_buy_base_volume', 'taker_buy_quote_volume', 'stop_loss_time','take_profit_time']]
        df.dropna(subset=subset,
                  inplace=True)
        self.df_null_values = df.isnull().sum()
        self.X = df.loc[:, cols]
        self.X_columns = cols
        if training:
            self.transformer = transformer
            self.transformer.fit(self.X)
        try:
            y = df.loc[:, 'real_class']
        except KeyError:
            df.rename(columns={"profitable": "real_class"}, inplace=True)
            y = df.loc[:, 'real_class']
        return self.transformer.transform(self.X), y

    def classify_model(self, config: ModelConfig, evaluate: bool = True):
        rf_random = RandomizedSearchCV(estimator=config.model_instance,
                                       param_distributions=config.params,
                                       n_iter=config.n_iter,
                                       cv=config.cv,
                                       verbose=config.verbose,
                                       random_state=config.random_state,
                                       n_jobs=config.n_jobs,
                                       scoring=config.scoring)
        rf_random.fit(self.X_train, self.y_train)

        self.best_random = rf_random.best_estimator_
        if evaluate:
            self.y_pred = self.best_random.predict(self.X_test)
            self.y_pred_train = self.best_random.predict(self.X_train)
            print(rf_random.best_params_)
            resumen_rs = pd.DataFrame(rf_random.cv_results_)
            print(resumen_rs)
            print("Classification Process is over")
            print(classification_report(self.label_encoder.inverse_transform(self.y_test),
                                        self.label_encoder.inverse_transform(self.y_pred)))

        accuracy = accuracy_score(self.y_test, self.y_pred)
        self.accuracy = accuracy
        print('accuracy: ', accuracy)

    def transform_predict(self, df: pd.DataFrame):
        labelled_df = triple_barrier_method(df, tp=self.tp, sl=self.sl, tl=self.tl, trade_cost=self.trade_cost)
        features_df = self.add_features(candles_df=labelled_df, external_feat=self.external_feat)
        X = self.column_transform(features_df=features_df, training=False)
        current_candle = X[-1].reshape(1, -1)
        return self.best_random.predict(current_candle)

    def analyse(self):
        ##############################
        ### Analyze best Model !!! ###
        ##############################

        import pandas as pd
        resumen_proba = pd.DataFrame()
        y_pred_transform = self.label_encoder.inverse_transform(self.y_pred)
        y_test_transform = self.label_encoder.inverse_transform(self.y_test)
        resumen_proba['y_test'] = y_test_transform
        resumen_proba['y_pred'] = y_pred_transform
        # resumen_proba[['0','1','2']]=pd.DataFrame(pred_prob)
        y_pred_train_transform = self.label_encoder.inverse_transform(self.y_pred_train)
        y_train_transform = self.label_encoder.inverse_transform(self.y_train)
        resumen_proba.to_csv('data/actual_predictions_' + self.model_kind + '.csv')

        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

        matrix = confusion_matrix(y_train_transform, y_pred_train_transform)
        matrix = matrix.astype('float') / matrix.sum(axis=1)[:, np.newaxis]

        print(classification_report(y_train_transform, y_pred_train_transform))
        print(classification_report(y_test_transform, y_pred_transform))

        print("Cohen - Kappa Score")
        print(cohen_kappa_score(self.y_test, self.y_pred))

        print("Matthews Correlation Coeff Score")
        print(matthews_corrcoef(self.y_test, self.y_pred))

        # Prepare DataFrames to store feature importances for each class
        gini_summary_list = []
        perm_summary_list = []
        print("Starting Gini Importance...")
        for i, class_name in enumerate(pd.Series(self.y).unique()):
            # Extract the classifier for the current class
            classifier = self.best_random.estimators_[i-1]
            
            # Gini Importance
            gini_importance = classifier.feature_importances_
            gini_df = pd.DataFrame({
                'Feature': self.X_columns,
                'Class': class_name,
                'Gini Importance': gini_importance
            })
            gini_summary_list.append(gini_df)
            
            # Permutation Importance
            result = permutation_importance(classifier, self.X, self.y == i, n_repeats=10, random_state=42, n_jobs=-1)
            perm_importances = result.importances_mean
            perm_df = pd.DataFrame({
                'Feature': self.X_columns,
                'Class': class_name,
                'Permutation Importance': perm_importances
            })
            perm_summary_list.append(perm_df)

        # Combine all class DataFrames into one
        gini_summary = pd.concat(gini_summary_list, axis=0)
        perm_summary = pd.concat(perm_summary_list, axis=0)

        gini_summary.sort_values("Gini Importance", ascending = False).to_csv('gini_feature_importance_with_class.csv', index=False)
        perm_summary.sort_values("Permutation Importance", ascending = False).to_csv('perm_feature_importance_with_class.csv', index=False)

        print("Gini feature importance summary with class saved to 'gini_feature_importance_with_class.csv'")
        print("Permutation feature importance summary with class saved to 'perm_feature_importance_with_class.csv'")
    
        # shap_summary_list = []
        # for i, class_name in enumerate(pd.Series(self.y).unique()):
        #     classifier = self.best_random.estimators_[i-1]
            
        #     explainer = shap.TreeExplainer(classifier)
        #     shap_values = explainer.shap_values(self.X)
        
        #     shap_importance = np.mean(np.abs(shap_values), axis=0)
        #     class_df = pd.DataFrame({
        #         'Feature': self.X_columns, 
        #         'Class': class_name,
        #         'SHAP Importance': shap_importance
        #     })
        #     shap_summary_list.append(class_df)
        # shap_summary = pd.concat(shap_summary_list, axis=0)
        # shap_summary.to_csv('shap_feature_importance_with_class.csv', index=False)

    # TODO: no encontre ningun usage
    def evaluate(self, df, model_path):
        # Transformación
        self.df = triple_barrier_method(df)
        self.df = self.add_features()
        model = pickle.load(open(model_path, 'rb'))
        self.y_pred = model.predict(self.df)
        print('y_pred-1:\n',self.y_pred-1)
        print('y_pred+1:\n', self.y_pred + 1)
        print('y_pred:\n', self.y_pred)
        print('\n\ny_test:\n',self.df.real_class)
        print(classification_report(self.df.real_class, self.y_pred-1))

    def eda(self, hypotesis=True):
        if self.visualize:
            # TODO: Agregar streamlit objects
            subplots, ax = plt.subplots(figsize=(15, 10))
            corr_matrix = sns.heatmap(self.df.corr(), annot=True, ax=ax)
        if hypotesis:
            df_cont = pd.DataFrame(columns=['column', 'fvalue/chi2', 'pvalue', 'type'])

            long = self.df[self.df['real_class'] == 1]
            nothing = self.df[self.df['real_class'] == 0]
            short = self.df[self.df['real_class'] == -1]

            for column in (self.to_scale + self.to_log):
                fvalue, pvalue = stats.f_oneway(short[column].dropna(),
                                                nothing[column].dropna(),
                                                long[column].dropna())
                df_cont.loc[len(df_cont)] = [column, round(fvalue, 3), pvalue, "numeric"]

            for column in self.categorical:
                contigency = pd.crosstab(self.df['real_class'], self.df[column])
                print(contigency, '\n')
                c, p, dof, expected = chi2_contingency(contigency)

                df_cont.loc[len(df_cont)] = [column, round(c, 3), p, "categoric"]
            # TODO Devolver pandas profiling o sweetviz
            print(df_cont.sort_values(by='pvalue', ascending=False))


if __name__ == "__main__":
    data = pd.read_csv("../../data/data/candles/test_candles.csv")
    external_features = {
        "close": {
            'macd': [[12, 24, 9]]
        }
    }
    tba = TripleBarrierAnalyser(df=data, external_feat=external_features)
    features_df = tba.prepare_data(data)
    feat_df = pd.read_csv("../../data/data/candles/test_features_df.csv")
    RF_CONFIG = ModelConfig(
        name="Random Forest",
        params={
            'estimator__n_estimators': [1000],  # Or use [int(x) for x in np.linspace(start=100, stop=1000, num=3)]
            'estimator__max_features': ['sqrt'],  # Or ['log2']
            'estimator__max_depth': [20, 55, 100],  # Or use [int(x) for x in np.linspace(10, 100, num=3)]
            'estimator__min_samples_split': [50, 100],
            'estimator__min_samples_leaf': [30, 50],
            'estimator__bootstrap': [True],
            'estimator__class_weight': ['balanced']
        },
        model_instance=RandomForestClassifier(),
        one_vs_rest=True,
        n_iter=10,
        cv=2,
        verbose=10
    )

    tba.transform_train(features_df=feat_df, model_config=RF_CONFIG)
    tba.analyse()
