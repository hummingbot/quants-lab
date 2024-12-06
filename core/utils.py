import glob
import os.path

import yaml


def dump_dict_to_yaml(config: dict, folder: str = ""):
    path = os.path.join(folder, config["id"] + ".yml")
    with open(path, 'w') as file:
        yaml.dump(config, file, sort_keys=False)


def load_dict_from_yaml(file_name: str, folder: str = ""):
    path = os.path.join(folder, file_name)
    with open(path, 'r') as file:
        data = yaml.safe_load(file)
    return data


def list_all_yaml_files(folder: str = ""):
    return [file for file in os.listdir(folder) if file.endswith(".yml")]


def load_all_yaml_files(folder: str = ""):
    data = []
    for file in list_all_yaml_files(folder):
        data.append(load_dict_from_yaml(os.path.join(folder, file)))
    return data

def list_all_files(folder: str = "", pattern: str = ""):
    return glob.glob(os.path.join(folder, pattern))