import importlib
import os
import pkgutil


def import_all_configs():
    current_dir = os.path.dirname(os.path.abspath(__file__))

    for package in pkgutil.walk_packages([current_dir]):
        if not package.ispkg:
            importlib.import_module(package.name)


import_all_configs()
