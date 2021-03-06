import argparse
import configparser

from train_setup import train

def preprocess_config(c):
    conf_dict = {}
    int_params = ['data.batch_size', 'data.episodes', 'data.gpu', 'data.cuda', 'train.epochs', 'train.patience']
    float_params = ['train.lr']
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict


parser = argparse.ArgumentParser(description='Run training')
parser.add_argument("--config", type=str, default="./src/df_v2/config/config_heart.conf",
                    help="Path to the config file.")

parser.add_argument("--data.dataset", type=str, default=None)
parser.add_argument("--data.split", type=str, default=None)
parser.add_argument("--data.batch_size", type=int, default=None)
parser.add_argument("--data.episodes", type=int, default=None)
parser.add_argument("--data.cuda", type=int, default=None)
parser.add_argument("--data.gpu", type=int, default=None)

parser.add_argument("--train.patience", type=int, default=None)
parser.add_argument("--train.lr", type=float, default=None)

# Run training
args = vars(parser.parse_args())
config = configparser.ConfigParser()
config.read(args['config'])
filtered_args = dict((k, v) for (k, v) in args.items() if not v is None)
config = preprocess_config({ **config['TRAIN'], **filtered_args })
train(config)
