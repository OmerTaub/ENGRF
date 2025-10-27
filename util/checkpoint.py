import os, torch

def save_ckpt(state: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_ckpt(path: str, map_location=None):
    return torch.load(path, map_location=map_location)