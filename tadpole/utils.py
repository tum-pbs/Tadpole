import logging,sys
from omegaconf import OmegaConf, DictConfig
from typing import Union
import logging,sys
import torch
from safetensors.torch import load_file

def read_config(config_path:Union[str,dict,DictConfig],resolve=True)-> dict:
    """
    Reads a configuration file and returns it as a dictionary.
    
    Args:
        config_path (str): Path to the configuration file.
        
    Returns:
        dict: Configuration data as a dictionary.
    """
    if isinstance(config_path, str):
        return OmegaConf.to_container(OmegaConf.load(config_path), resolve=resolve)
    elif isinstance(config_path, DictConfig):
        return OmegaConf.to_container(config_path, resolve=True)
    elif isinstance(config_path, dict):
        return config_path
    else:
        raise TypeError(f"Unsupported type for config_path: {type(config_path)}")

def get_logger(logger=None,debug=False):
    if logger is not None:
        return logger
    logger = logging.getLogger(__name__)
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def load_weights(weight:Union[str, dict],model_name:str) -> dict:
    if isinstance(weight, str):
        print(f"Loading pretrained weights for {model_name} from {weight}.")
        if weight.endswith(".safetensors"):
            weight = load_file(weight, device="cpu")
        else:
            weight = torch.load(weight, weights_only=True)
    else:
        print(f"Using provided weights for {model_name}.")
    return weight