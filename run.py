'''
Builds and trains/tests a model based on input parameters, which are specified
via a YAML configuration file and optional command line arguments.

Example usage:
    - Train the model found in experiments/example/config_1.yaml
        python main.py --mode train --config example/config_1

    - Test the model found in lightning_logs/example/config_1/version_0/
        python main.py --mode test --config example/config_1/version_0
'''

import argparse
import os
import yaml

from core import train, test

#Parse args
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default=None, help='train or test.')
parser.add_argument('--config', type=str, default=None, help='YAML configuration file path relative to ./experiments with or without extension.')

args = vars(parser.parse_known_args()[0])
mode = args['mode']
config_path = args['config']

#Load config
if config_path != None:

    if mode == 'train':
        config_file = os.path.join('experiments/', config_path+'.yaml')
    elif mode == 'test':
        config_file = os.path.join('lightning_logs/', config_path, 'config.yaml')
    else:
        raise ValueError(f"Mode {mode} is invalid.")

    try:
        #Open YAML file
        with open(config_file, "r") as file:
            config = yaml.safe_load(file)

    except Exception as e:
        raise ValueError(f"Config {config_path} is invalid.")

else:
    raise ValueError("An experiment configuration file must be provided.")

#Trainer args
trainer_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
trainer_parser.add_argument('--default_root_dir', type=str)
trainer_parser.add_argument('--max_time', type=str)

#Look for trainer CL arguments
config['train'].update(vars(trainer_parser.parse_known_args()[0]))

#Data args
data_parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
data_parser.add_argument('--data_dir', type=str)

#Look for data CL arguments
config['data'].update(vars(data_parser.parse_known_args()[0]))

#Train
if mode == 'train':
    train(config_path, config)

#Test
elif mode == 'test':
    log_dir = os.path.join('lightning_logs/', config_path)

    test(log_dir, config)
