import torch.nn.functional as F

def fm_loss(v_pred, delta):
    return F.mse_loss(v_pred, delta)

def gfm_target(dt_h, Dh_delta):
    return dt_h + Dh_delta

def gfm_loss(v_tilde_pred, target):
    return F.mse_loss(v_tilde_pred, target)