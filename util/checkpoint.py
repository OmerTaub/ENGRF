import os, torch

def save_ckpt(path: str,
              model_state_dict: dict, 
              optimizer_state_dict: dict = None, 
              global_step: int = None, 
              epoch: int = None,
              **kwargs: dict,
              ):
    """
    path (required)
    model_state_dict (required)
    optimizer_state_dict (optional)
    global_step (optional)
    epoch (optional)
    config (optional)
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    save_dict = {
        "model_state_dict": model_state_dict,
    }

    if optimizer_state_dict is not None:
        save_dict["optimizer_state_dict"] = optimizer_state_dict
    if global_step is not None:
        save_dict["global_step"] = global_step
    if epoch is not None:
        save_dict["epoch"] = epoch
    for key, value in kwargs.items():
        save_dict[key] = value
    torch.save(save_dict, path)

def load_ckpt(path: str, map_location=None):
    return torch.load(path, map_location=map_location)


def get_outdir(cfg: dict, resume: bool = False) -> str:
    out_dir = cfg.get("experiment", {}).get("out_dir", "runs")
    os.makedirs(out_dir, exist_ok=True)
    
    runs = [int(run.split("_")[-1]) for run in os.listdir(out_dir) if run.startswith("run_")]
    latest_run = 0 if len(runs) == 0 else max(runs)
    if resume:
        return os.path.join(out_dir, f"run_{latest_run}")
    else:
        return os.path.join(out_dir, f"run_{latest_run + 1}")


def find_latest_checkpoint(ckpt_dir: str) -> str:
    """
    assumes ckpts are saved in the format: "best_stage[0-2]_ep[000-999].pt"
    """
    if not os.path.exists(ckpt_dir):
        return None
    
    checkpoints = [f for f in os.listdir(ckpt_dir) if f.endswith(".pt") and "_ep" in f]
    if not checkpoints:
        return None
    
    # Sort by epoch number extracted from filename
    latest_ckpt = max(checkpoints, key=lambda x: int(x.split("_ep")[1].split(".pt")[0]))
    return os.path.join(ckpt_dir, latest_ckpt)



def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, map_location=None):
    """
    assumes checkpoints contains:
    - state_dict (required)
    - opt_state_dict (optional)

    returns:
    - model: loaded model
    - optimizer: loaded optimizer
    - global_step: global step of checkpoint
    - epoch: epoch of checkpoint
    """
    if not os.path.exists(path):
        return None, model, optimizer, 0, 1

    ckpt = torch.load(path, map_location=map_location)
    state = ckpt.get("state_dict", None)
    if state is None:
        raise ValueError(f"Checkpoint {path} does not contain a state_dict")
    
    opt_state = ckpt.get("opt_state_dict", None)
    if opt_state is not None:
        try:
            optimizer.load_state_dict(opt_state, map_location=map_location)
        except Exception as e:
            logger.warning(f"[load_checkpoint] Failed to load optimizer state: {e}")

    global_step = ckpt.get("global_step", 0)
    epoch = ckpt.get("epoch", None)

    # if no "epoch" in checkpoint, try to extract epoch from filename, if that fails set to 1
    if epoch is None:
        try:
            epoch = int(os.path.basename(path).split("_ep")[1].split(".pt")[0])
        except Exception as e:
            logger.warning(f"[load_checkpoint] Failed to extract epoch from filename: {e}")
            epoch = 1
    
    return model, optimizer, global_step, epoch
    

def load_latest_checkpoint(ckpt_dir:str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, map_location=None):
    """
    returns:
    - latest_ckpt: path to latest checkpoint
    - model: loaded model
    - optimizer: loaded optimizer
    - global_step: global step of latest checkpoint
    - epoch: epoch of latest checkpoint
    """
    latest_ckpt = find_latest_checkpoint(ckpt_dir)
    if latest_ckpt is None:
        return None, model, optimizer, 0, 1
    
    model, optimizer, global_step, epoch = load_checkpoint(latest_ckpt, model, optimizer, map_location)
    return latest_ckpt, model, optimizer, global_step, epoch