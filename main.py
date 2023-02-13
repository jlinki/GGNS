import yaml
import argparse


def main(config=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="default", help="the config.yaml specified by this argument will be used")
    args = parser.parse_args()

    with open("config/" + args.config + ".yaml", 'r') as file:
        try:
            config = yaml.safe_load(file)
            wandb_log = config.get("wandb_log")
            imputation = config.get("imputation")

            # start training using the loaded config
            if imputation == "lstm":
                from src.algorithms.lstm_train import train
                train(config, wandb_log)
            else:
                from src.algorithms.standard_train import train
                train(config, wandb_log)
        except yaml.YAMLError as exc:
            print(exc)


if __name__ == '__main__':
    main()
