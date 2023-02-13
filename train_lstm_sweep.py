import yaml
from src.algorithms.lstm_train import train

def main(config=None):
    try:
        with open("config/default_lstm.yaml", 'r') as file:
            try:
                config = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)
    except FileNotFoundError:
        pass
    train(config, wandb_log=True)


if __name__ == '__main__':
    main()
