import os
import time
import torch
import yaml
import wandb

from util.Types import *


def wandb_init(config: Dict, wandb_log: bool, root_dir: str) -> Tuple:
    """
    Initializes the wandb run and defines the config and folder for the current run
    Args:
        config: Dictionary containing all parameters to initialize the wandb run
        wandb_log: True if wandb logs the run
        root_dir: Current working directory
    Returns: Tuple.
        config: Potentially updated config (e.g. if a wandb sweep is performed)
        run_folder_dir: Folder of the current run
    """
    if wandb_log:
        wandb.init(config=config, project=config.get("project"))
        config = wandb.config
        print(config)
        run_folder_name = wandb.run.name
        if wandb.run.sweep_id is not None:
            project_name = wandb.run.get_project_url().partition("jlinki/")[-1]
            run_folder_dir = os.path.join(root_dir, "data", project_name, "sweeps", wandb.run.sweep_id, run_folder_name)
        else:
            run_folder_name = run_folder_name + "_" + time.strftime('%d-%m-%Y_%H_%M_%S', time.localtime())
            run_folder_dir = os.path.join(root_dir, "data", wandb.config.project, "runs", run_folder_name)
        if not os.path.exists(run_folder_dir):
            os.makedirs(run_folder_dir)
        config_file_path = os.path.join(wandb.run.dir, 'config.yaml')
        with open(config_file_path, 'r') as file:
            config_dict = yaml.safe_load(file)  # saves the config for later evaluation with the correct parameters
        with open(os.path.join(run_folder_dir, "config.yaml"), 'w') as yaml_file:
            yaml.dump(config_dict, yaml_file)
    else:
        run_folder_dir = None
    return config, run_folder_dir


def wandb_loss_logger(wandb_log: bool, loss: float, name: str, epoch: int, best: float, model, save_path: str) -> float:
    """
    Logs a loss of a certain epoch using wandb.
    If the new loss is lower than current best, the best is updated and the corresponding parameters are saved.
    Args:
        wandb_log: True if wandb logs the run
        loss: Current value of the loss
        name: Name of the current loss
        epoch: Current epoch
        best: Best loss of that name so far
        model: GNN to evaluate
        save_path: Path where the model parameters for best losses should be saved

    Returns:
        best: New best loss
    """
    if wandb_log:
        wandb.log({name: loss, "epoch": epoch})
    if loss < best:
        if wandb_log:
            wandb.run.summary["best " + name] = loss
            wandb.run.summary["epoch best " + name] = epoch
            torch.save(model.state_dict(), os.path.join(save_path, "GNN_model_best_" + name.replace(" ", "_").replace("/", "_")))
        best = loss
    return best


def wandb_config_update(wandb_log: bool, update: Dict):
    """
    Updates the wandb config with the dict of update
    Args:
        wandb_log: True if wandb is used
        update: Dict to add to config
    """
    if wandb_log:
        wandb.config.update(update)
    else:
        pass
