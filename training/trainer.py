# import os, torch
# from torch.utils.data import DataLoader
# from .losses import fm_loss, gfm_loss, gfm_target
# from models.posterior_mean import PosteriorMean
# from models.rectified_flow import RectifiedFlow
# from models.gauge import GaugeField, GaugeFlow

# def sample_t(batch, device):
#     return torch.rand(batch, 1, 1, 1, device=device)

# def y_embed_default(y):  # use degraded input as measurement embedding
#     return y

# class ENGRFAbs(torch.nn.Module):
#     def __init__(self, cfg):
#         super().__init__()
#         self.cfg = cfg
#         mcfg = cfg['model']
#         self.pm = PosteriorMean(mcfg['posterior_mean'], mcfg['pm_unet'])
#         self.rf = RectifiedFlow(**mcfg['rf_unet'])
#         self.W = GaugeField(**mcfg['gauge_field'])
#         self.hflow = GaugeFlow(self.W, **mcfg['gauge_flow'])


#     def forward(self, batch, stage):
#         x = batch['x'].to(next(self.parameters()).device)  # (B,1,H,W)
#         y = batch['y'].to(x.device)

#         B = x.size(0)
#         t = sample_t(B, x.device)

#         x_star = self.pm(y)  # posterior-mean estimate from degraded input
#         Z_t = (1.0 - t) * x_star + t * x
#         delta = x - x_star

#         if stage == 1:
#             v_pred = self.pm(Z_t, t)
#             loss = fm_loss(v_pred, delta)
#             return loss, {'fm_loss': loss.item()}

#         # Stage 2: Gauge-FM
#         if stage == 2:
#             y_embed = y_embed_default(y)
#             Z_t_tilde = self.hflow.h(Z_t, t, y_embed)
#             dt_h = self.hflow.dt_h(Z_t, t, y_embed)
#             Dh_delta = self.hflow.jvp_h(Z_t, t, y_embed, delta)
#             target = gfm_target(dt_h, Dh_delta)

#             # conjugated velocity prediction at gauged point:
#             Z_pre = self.hflow.h_inv(Z_t_tilde, t, y_embed)
#             v_base = self.rf(Z_pre, t)
#             dt_h_at_x = self.hflow.dt_h(Z_t_tilde, t, y_embed)
#             Dh_v = self.hflow.jvp_h(Z_t_tilde, t, y_embed, v_base)
#             v_tilde_pred = dt_h_at_x + Dh_v

#             loss = gfm_loss(v_tilde_pred, target)
#             return loss, {'gfm_loss': loss.item()}

# def train_loop(cfg, train_ds, val_ds, device='cuda'):
#     model = ENGRFAbs(cfg).to(device)

#     # Robustly coerce numeric hyperparams (handles "1e-4" strings, etc.)
#     lr = float(cfg['train']['lr'])
#     wd = float(cfg['train']['weight_decay'])
#     batch_size = int(cfg['train']['batch_size'])
#     num_workers = int(cfg['train']['num_workers'])

#     opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
#     train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
#     val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

#     # Stage 1
#     for ep in range(cfg['train']['epochs_stage1']):
#         model.train()
#         for i,b in enumerate(train_loader):
#             opt.zero_grad()
#             loss, logs = model(i, stage=1)
#             loss.backward(); opt.step()
#             if i % cfg['train']['log_interval'] == 0:
#                 print(f"[Stage1][Ep {ep+1}] it={i} fm_loss={logs['fm_loss']:.6f}")

#     # Stage 2
#     for ep in range(cfg['train']['epochs_stage2']):
#         model.train()
#         for i,b in enumerate(train_loader,1):
#             opt.zero_grad()
#             loss, logs = model(b, stage=2)
#             loss.backward(); opt.step()
#             if i % cfg['train']['log_interval'] == 0:
#                 print(f"[Stage2][Ep {ep+1}] it={i} gfm_loss={logs['gfm_loss']:.6f}")

#     return model