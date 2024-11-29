import pickle
import os
import pandas as pd


def load_pickle(file_name: str, root_path: str = ""):
    with open(os.path.join(root_path, "data", "models", file_name), "rb") as f:
        return pickle.load(f)


def load_all_pickles(root_path: str = ""):
    models_path = os.path.join(root_path, "data", "models")
    files = os.listdir(models_path)
    models = {}
    for file in files:
        if file != ".gitignore":
            models[file] = load_pickle(file, root_path)
    return models


def generate_pickle_report(root_path: str = ""):
    models = load_all_pickles(root_path)
    parsed_data = []
    for model_name, model_details in models.items():
        classification_report = model_details['classification_report']
        parsed_data.append({
            'model_name': model_name,
            'connector_name': model_details['connector_name'],
            'trading_pair': model_details['trading_pair'],
            'tp': model_details['tp'],
            'short_precision': classification_report['-1.0']['precision'],
            'short_recall': classification_report['-1.0']['recall'],
            'short_f1_score': classification_report['-1.0']['f1-score'],
            'short_support': classification_report['-1.0']['support'],
            'long_precision': classification_report['1.0']['precision'],
            'long_recall': classification_report['1.0']['recall'],
            'long_f1_score': classification_report['1.0']['f1-score'],
            'long_support': classification_report['1.0']['support'],
            'accuracy': classification_report['accuracy'],
            'macro_avg_precision': classification_report['macro avg']['precision'],
            'macro_avg_recall': classification_report['macro avg']['recall'],
            'macro_avg_f1_score': classification_report['macro avg']['f1-score'],
            'macro_avg_support': classification_report['macro avg']['support'],
            'weighted_avg_precision': classification_report['weighted avg']['precision'],
            'weighted_avg_recall': classification_report['weighted avg']['recall'],
            'weighted_avg_f1_score': classification_report['weighted avg']['f1-score'],
            'weighted_avg_support': classification_report['weighted avg']['support'],
            'extra_features': model_details['extra_features']
        })
    df = pd.DataFrame(parsed_data)
    return df
