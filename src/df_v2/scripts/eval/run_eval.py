import argparse
import configparser

from eval_setup import eval

parser = argparse.ArgumentParser(description="Run evaluation")

def preprocess_config(c):
    conf_dict = {}
    int_params = ["data.batch_size", "data.episodes", "data.gpu", "data.cuda"]
    float_params = ["data.train_size", "data.test_size"]
    for param in c:
        if param in int_params:
            conf_dict[param] = int(c[param])
        elif param in float_params:
            conf_dict[param] = float(c[param])
        else:
            conf_dict[param] = c[param]
    return conf_dict


parser = argparse.ArgumentParser(description="Run evaluation")
parser.add_argument("--config", type=str, default="./src/df_v1/config/config_heart.conf",
                    help="Path to the config file.")

parser.add_argument("--data.dataset", type=str, default=None)
parser.add_argument("--data.split", type=str, default=None)
parser.add_argument("--data.batch_size", type=int, default=None)
parser.add_argument("--data.episodes", type=int, default=None)
parser.add_argument("--data.cuda", type=int, default=None)
parser.add_argument("--data.gpu", type=int, default=None)

parser.add_argument("--data.train_size", type=float, default=None)
parser.add_argument("--data.test_size", type=float, default=None)

parser.add_argument("--model.path", type=str, default=None)
parser.add_argument("--model.json_save_path", type=str, default=None)
parser.add_argument("--model.weights_save_path", type=str, default=None)

# Run test
args = vars(parser.parse_args())
config = configparser.ConfigParser()
config.read(args["config"])
filtered_args = dict((k, v) for (k, v) in args.items() if not v is None)
config = preprocess_config({ **config["EVAL"], **filtered_args })
eval(config)