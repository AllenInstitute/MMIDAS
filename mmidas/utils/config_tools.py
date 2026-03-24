import numpy as np
import scipy.io as sio
import toml
from pathlib import Path


def get_paths(toml_file, sub_file=['data', 'models'], verbose=False):
    """Loads dictionary with path names and any other variables set through xxx.toml

    Args:
        verbose (bool, optional): print paths

    Returns:
        config: dict
    """

    # package_dir = Path().resolve() #.parents[0]
    # Get the directory of the current script file
    package_dir = Path(__file__).resolve().parent.parent.parent
    config_file = package_dir / toml_file
    print(config_file)

    if not Path(config_file).is_file():
        print(f'Did not find project`s toml file: {config_file}')

    f = open(config_file, "r")
    config = toml.load(f)
    f.close()

    config['paths'].update({'main_dir': package_dir})

    if verbose:
        for key in config.keys():
            print(f'{key}: {config[key]}')

    for key in config:
        if key=='paths':
            for key2 in config['paths']:
                if Path(config['paths'][key2]).exists():
                    config['paths'][key2] = Path(config['paths'][key2])
        if key in sub_file:
            print(f'Getting files directories belong to {key}...')
            for key2 in config[key]:
                if Path(config[key][key2]).exists():
                    config[key][key2] = Path(config[key][key2])

    return config


def print_attrs(name, obj):
    print(name)
    for key, val in obj.attrs.items():
        print("    %s: %s" % (key, val))
    return

