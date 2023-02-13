import os
import numpy as np
import yaml

from util.Types import *
from AbstractArchitecture import AbstractArchitecture


def create_path(path: str) -> None:
    if not os.path.exists(path):
        from pathlib import Path
        Path(path).mkdir(parents=True, exist_ok=True)


def save_dict(path: str, file_name: str, dict_to_save: Dict, overwrite: bool = False, save_type: str = "npz") -> None:
    """
    Saves the given dict
    Args:
        path: Full/Absolute path to save to
        file_name: Name of the file to save the dictionary to
        dict_to_save: The dictionary to save
        overwrite: Whether to overwrite an existing file
        save_type: The type to save the dict as

    Returns:

    """
    if not file_name.endswith(f".{save_type}"):
        file_name += f".{save_type}"
    create_path(path)
    file_to_save = os.path.join(path, file_name)
    if overwrite or not os.path.isfile(file_to_save):
        if save_type == "npz":
            np.savez_compressed(file_to_save, **dict_to_save)
        elif save_type == "yaml":
            yaml.dump(dict_to_save, file_name, sort_keys=True, indent=2)
        else:
            raise ValueError(f"Unknown save_type '{save_type}' for dictionary")


def load_dict(path: str) -> dict:
    """
    Loads the dictionary saved at the specified path
    Args:
        path: Path to the npz file in which the dictionary is saved. May or may not include the ".npz" at the end

    Returns: The dictionary saved at the specified path
    """
    if not path.endswith(".npz"):
        path += ".npz"
    assert os.path.isfile(path), "Path '{}' does not lead to a .npz file".format(path)
    return dict(np.load(path, allow_pickle=True))


def load_architecture(state_dict_path: str, network_config: Union[dict, str]) -> AbstractArchitecture:
    """

    Args:
        state_dict_path: Path to the state_dict saved by torch.save
        network_config: Config to instantiate the discriminator with. Can be either a dictionary, or a path to a
        .npz file containing a dictionary

    Returns: The discriminator with the appropriate state_dict, i.e., the correct network with loaded weights

    """
    if isinstance(network_config, str):
        network_config = load_dict(network_config)

    network_type = network_config.get("type").item()
    del network_config["type"]
    assert issubclass(network_type,
                      AbstractArchitecture), "Must inherit from Network base class, given {} of type {}".format(
        network_type, network_type)

    network_config = undo_numpy_conversions(network_config)
    network_config["input_shape"] = tuple(network_config.get("input_shape"))
    discriminator = network_type(**network_config)
    discriminator.load(load_path=state_dict_path)
    return discriminator


def undo_numpy_conversions(dictionary: dict) -> dict:
    """
    Numpy does some weird conversions when you save dictionary objects. This method undoes them.
    Args:
        dictionary: The dictionary to un-convert

    Returns: The same dictionary but with un-numpied values

    """
    converted_dictionary = dictionary
    for converting_type in [float, int, dict]:
        converted_dictionary = {
            k: v.item() if isinstance(v, np.ndarray)
                           and (v.ndim == 0 or len(v) == 1)
                           and isinstance(v.item(), converting_type)
            else v for k, v in converted_dictionary.items()
        }

    none_converted_dict = {k: None if isinstance(v, np.ndarray)
                                      and (v.ndim == 0 or len(v) == 1)
                                      and v == None else v
                           for k, v in converted_dictionary.items()}
    tuple_converted_dict = {k: (v,) if isinstance(v, np.ndarray) and
                                       (v.ndim == 0 or len(v) == 1) and
                                       isinstance(v.item(), int) else v
                            for k, v in none_converted_dict.items()}
    return tuple_converted_dict
