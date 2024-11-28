import pickle
import random

import numpy as np
import pandas as pd
import shap
from features import Features
from scipy import stats
from scipy.stats import chi2_contingency
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance

# import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    make_scorer,
    matthews_corrcoef,
    precision_score,
)

# from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler


def get_random_matrix(rows = None, ):
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

class TripleBarrierAnalizer:
    def __init__(self, df = pd.DataFrame(), features_dict={}, external_feat={}, make_pickle=True,
                 tp = 0.001, sl = 0.001, tl = 3600, trade_cost = 0.0006,
                 n_estimators=[1000], max_depth=[20, 30, 100],
                 min_samples_split=[50, 100],
                 min_samples_leaf=[30, 50],
                 class_weight=['balanced'],
                 model_metric = 'accuracy',
                 multi_classifier = [0.01,0.01]


                 
                 ):
        self.trgt = None
        self.labelled_df = None
        self.df = df
        self.transformer = None
        self.classes = None
        self.X = None
        self.y = None,
        self.X_train = None,
        self.X_test = None,
        self.y_train = None,
        self.y_test = None,
        self.X_columns = [],
        self.visualize = False,
        self.to_log = None,
        self.categorical = None,
        self.to_scale = None,
        self.results = pd.DataFrame()
        self.features_dict = features_dict
        self.external_feat = external_feat
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.class_weight = class_weight
        self.model_metric = model_metric
        self.tp = tp
        self.sl = sl
        self.tl = tl
        self.multi_classifier = multi_classifier

        if len(df) != 0:
            self.df = self.apply_triple_barrier_method(df, tp = self.tp, sl = self.sl, tl = self.tl, trade_cost = trade_cost)
            self.df = self.add_features(self.external_feat)
        self.make_pickle=make_pickle

    def transform_train(self,
                        model: str
                        ):
        self.model = model
        self.model_pre_processing()
        self.classify_model(model_kind=model,
                            n_estimators = self.n_estimators,
                            min_samples_split = self.min_samples_split,
                            min_samples_leaf = self.min_samples_leaf,
                            max_depth = self.max_depth,
                            class_weight = self.class_weight
            )

        if self.make_pickle:
            data_to_pickle = {
                'pipeline': Pipeline([
                    ('transformer', self.transformer),
                    ('prediction', self.best_random)
                ]),
                'extra_features': self.external_feat
            }
            with open('pipeline.pkl', 'wb') as f:
                pickle.dump(data_to_pickle, f)

    def transform_predict(self, df):
        self.df = df
        self.df = self.add_features()

        X = self.column_transform(training=False)
        current_candle = X[-1].reshape(1, -1)
        return self.best_random.predict(current_candle)

    def evaluate(self, df, model_path):
        # Transformación
        self.df = df
        self.apply_triple_barrier_method()
        self.df = self.add_features()
        model = pickle.load(open(model_path, 'rb'))
        self.y_pred = model.predict(self.df)
        print('y_pred-1:\n',self.y_pred-1)
        print('y_pred+1:\n', self.y_pred + 1)
        print('y_pred:\n', self.y_pred)
        print('\n\ny_test:\n',self.df.real_class)
        print(classification_report(self.df.real_class, self.y_pred-1))

    def apply_triple_barrier_method(self, df, tp=0.01, sl=0.01, tl=3600, trade_cost=0.0006):
        df.index = pd.to_datetime(df.timestamp, unit="ms")
        if "target" not in df.columns:
            # df["target"] = 0.0003
            df["target"] = df["close"].rolling(1000).std() / df["close"]
        df["tl"] = df.index + pd.Timedelta(seconds=tl)
        df.dropna(subset="target", inplace=True)

        try:
            df = self.apply_tp_sl_on_tl(df, tp=tp, sl=sl)
        except:
            df['signal'] = 1
            df = self.apply_tp_sl_on_tl(df, tp=tp, sl=sl)

        df = self.get_bins(df, trade_cost)

        df["tp"] = df["target"] * tp
        df["sl"] = df["target"] * sl
        df['profitable'] = np.where(df['net_pnl'].between(-self.multi_classifier[0],-self.multi_classifier[1]), 0, df['profitable'])

        df["take_profit_price"] = df["close"] * (1 + df["tp"] * df["signal"])
        df["stop_loss_price"] = df["close"] * (1 - df["sl"] * df["signal"])
        df.to_csv('data/check_triple_barrier.csv')
        return df

    @staticmethod
    def get_bins(df, trade_cost):
        # 1) prices aligned with events
        px = df.index.union(df["tl"].values).drop_duplicates()
        px = df.close.reindex(px, method="ffill")
        # 2) create out object
        df["trade_pnl"] = (px.loc[df["close_time"].values].values / px.loc[df.index] - 1) * df["signal"]
        df["net_pnl"] = df["trade_pnl"] - trade_cost
        df["profitable"] = np.sign(df["trade_pnl"] - trade_cost)
        df["close_price"] = px.loc[df["close_time"].values].values
        return df

    @staticmethod
    def apply_tp_sl_on_tl(df: pd.DataFrame, tp: float, sl: float):
        events = df[df["signal"] != 0].copy()
        if tp > 0:
            take_profit = tp * events["target"]
        else:
            take_profit = pd.Series(index=df.index)  # NaNs
        if sl > 0:
            stop_loss = - sl * events["target"]
        else:
            stop_loss = pd.Series(index=df.index)  # NaNs

        for loc, tl in events["tl"].fillna(df.index[-1]).items():
            df0 = df.close[loc:tl]  # path prices
            df0 = (df0 / df.close[loc] - 1) * events.at[loc, "signal"]  # path returns
            df.loc[loc, "stop_loss_time"] = df0[df0 < stop_loss[loc]].index.min()  # earliest stop loss.
            df.loc[loc, "take_profit_time"] = df0[df0 > take_profit[loc]].index.min()  # earliest profit taking.
        df["close_time"] = df[["tl", "take_profit_time", "stop_loss_time"]].dropna(how="all").min(axis=1)
        df["close_type"] = df[["take_profit_time", "stop_loss_time", "tl"]].dropna(how="all").idxmin(axis=1)
        df["close_type"].replace({"take_profit_time": "tp", "stop_loss_time": "sl"}, inplace=True)
        return df

    def add_features(self, external_feat={}):
        ft = Features(external_feat, self.df)
        self.df = self.df.set_index('timestamp')
        return ft.add_features()

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

    def column_transform(self, training=True):
        # scale_pipe = make_pipeline(StandardScaler())
        # log_pipe = make_pipeline(PowerTransformer())
        # categorical_pipe = make_pipeline(OneHotEncoder(sparse=False, handle_unknown="ignore"))

        # self.to_log = ['volume', 'MACD_12_26_9_ALPHA', 'RSI_14_MEMO']
        # self.categorical = ['MACD_12_26_9_SIDE', 'RSI_14_SIDE', 'MACD_12_26_9_CROSS']
        # self.to_scale = ['MACD_12_26_9', 'MACDs_12_26_9', 'MACD_12_26_9_VALUE', 'RSI_14']
        # cols = self.categorical + self.to_log + self.to_scale

        cols = self.df.select_dtypes(include=['int', 'float', 'bool'])
        cols = [column for column in cols if column not in ['ret_2','tp','sl', 't1', 'Pt1', 'ret',
                'stop_loss_time', 'take_profit_time', 'close_price','index','tl',
                'ret/trgt', 'time','close_time','real_class','marker','index','Unnamed: 0', 'ignore',"datetime",
                    'trgt','open','volume','close','taker_base_vol','qav','low','high','taker_quote_vol','num_trades',
                    'trade_pnl', 'net_pnl','real_class','profitable','signal','take_profit_price','stop_loss_price','timestamp']]
        print(self.df.dtypes)
        transformer = ColumnTransformer(
            transformers=[
                # ("subset", lambda x: x[cols], self.df),
                # ("scale", scale_pipe, self.to_scale),
                # ("log_transform", log_pipe, self.to_log),
                # ("oh_encode", categorical_pipe, self.categorical)
            ],
            remainder='passthrough'
        )
        self.df.dropna(subset = [c for c in self.df.columns if c not in ['stop_loss_time','take_profit_time']], inplace = True)
        self.df.isnull().sum().to_excel("df_null_values.xlsx")
        self.X = self.df.loc[:, cols]
        self.X_columns = cols
        self.X.to_excel("data/X_input_pre_process.xlsx", index = False)
        # self.X.to_excel('model_X_entrada.xlsx')
        if training:
            self.transformer = transformer
            self.transformer.fit(self.X)
        # self.df.to_excel("data/self_df.xlsx")
        return self.transformer.transform(self.X)


    def model_pre_processing(self, admitted_cols=[], test_size=0.2):
        self.X = self.column_transform()
        pd.DataFrame(self.X).to_excel('model_input_X.xlsx')
        if len(admitted_cols) != 0:
            self.df = self.df[admitted_cols]
        try:
            y = self.df.loc[:, 'real_class']
        except:
            self.df.rename(columns={"profitable":"real_class"}, inplace=True)
            y = self.df.loc[:, 'real_class']
        from sklearn.preprocessing import LabelEncoder
        label_encoder = LabelEncoder()
        label_encoder.fit(y)
        y = label_encoder.transform(y)
        self.classes = label_encoder.classes_
        self.label_encoder = label_encoder

        self.y = y
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, test_size=test_size)

    def classify_model(self,
                       evaluation=True,
                       # n_estimators=[int(x) for x in np.linspace(start=100, stop=1000, num=3)],
                       n_estimators=[1000],
                       max_features=['sqrt'],
                       # n_estimators = [796],
                       # max_features=['log2'],
                       # max_depth=[int(x) for x in np.linspace(10, 100, num=3)],
                       max_depth = [20,55,100],
                       min_samples_split=[50, 100],
                       min_samples_leaf=[30,50],
                       # min_samples_leaf=[50],
                       bootstrap=[True],
                       model_kind="RF Classifier",
                       penalty=['l1', 'l2', 'elasticnet'],
                       dual=[False, True],
                       tol=[1e-4, 1e-3, 1e-2, 1e-1],
                       c=np.logspace(-4, 4, 20),
                       fit_intercept=[True, False],
                       intercept_scaling=[0.001, 0.01, 0.1, 1, 10, 100],
                       class_weight=['balanced'],
                       solver=['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
                       max_iter=[100, 200, 300, 400, 500],
                       warm_start=[False, True],
                       n_jobs=[-1],
                       n_iter=100,
                       learning_rate=[0.001, 0.01, 0.1, 1, 10],
                       algorithm=['SAMME', 'SAMME.R'],
                       booster=['gbtree', 'gblinear', 'dart'],
                       min_child_weight=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                       subsample=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       colsample_bytree=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       colsample_bylevel=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       colsample_bynode=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                       reg_alpha=[0, 0.001, 0.01, 0.1, 1, 10, 100],
                       reg_lambda=[0, 0.001, 0.01, 0.1, 1, 10, 100],
                       one_vs_rest=True

                       ):

        if model_kind == "RF Classifier":
            random_grid = {'estimator__n_estimators': n_estimators,
                           'estimator__max_features': max_features,
                           'estimator__max_depth': max_depth,
                           'estimator__min_samples_split': min_samples_split,
                           'estimator__min_samples_leaf': min_samples_leaf,
                           'estimator__bootstrap': bootstrap,
                           'estimator__class_weight': class_weight}
            if one_vs_rest:
                model = OneVsRestClassifier(RandomForestClassifier(class_weight='balanced', random_state=42))
            else:
                model=RandomForestClassifier()
        elif model_kind == "LR":
            random_grid = {
                'estimator__penalty': penalty,
                'estimator__dual': dual,
                'estimator__tol': tol,
                'estimator__C': c,
                'estimator__fit_intercept': fit_intercept,
                'estimator__intercept_scaling': intercept_scaling,
                'estimator__class_weight': class_weight,
                'estimator__solver': solver,
                'estimator__max_iter': max_iter,
                'estimator__warm_start': warm_start,
                'estimator__n_jobs': n_jobs
            }
            model = OneVsRestClassifier(LogisticRegression())
        elif model_kind == "AdaBoost":
            random_grid = {
                'estimator__learning_rate': learning_rate,
                'estimator__algorithm': algorithm
            }
            model = OneVsRestClassifier(AdaBoostClassifier())
        elif model_kind == "XGBoost":
            random_grid = {
                'estimator__booster': booster,
                'estimator__n_estimators': n_estimators,
                'estimator__learning_rate': learning_rate,
                'estimator__max_depth': max_depth,
                'estimator__min_child_weight': min_child_weight,
                'estimator__subsample': subsample,
                'estimator__colsample_bytree': colsample_bytree,
                'estimator__colsample_bylevel': colsample_bylevel,
                'estimator__colsample_bynode': colsample_bynode,
                'estimator__reg_alpha': reg_alpha,
                'estimator__reg_lambda': reg_lambda
            }
            model = OneVsRestClassifier(XGBClassifier())

        self.model_kind = model_kind
        print(self.model_metric)
        rf_random = RandomizedSearchCV(estimator=model, param_distributions=random_grid, n_iter=n_iter, cv=4, verbose=10,
                                       random_state=42, n_jobs=-1, scoring=self.model_metric)

        rf_random.fit(self.X_train, self.y_train)

        self.best_random = rf_random.best_estimator_
        if evaluation:
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

