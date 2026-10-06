"""Differentiable SS silhouette loss in the provided target camera.

SS has no color features. Decode predicted/GT SS and ray march occupancy;
the target is the GT geometry silhouette, not a background-thresholded RGB.
Grid convention is [x,y,z], world cube [-.5,.5]^3, OpenCV camera.
"""
import torch
import torch.nn.functional as F

from .render_loss import RenderLossConfig, load_render_camera, schedule_lambda_render, predict_clean_features


def render_occupancy(volume, extrinsics, intrinsics, resolution, samples=96):
    device = volume.device
    uv = (torch.arange(resolution, device=device, dtype=torch.float32)+0.5)/resolution
    y, x = torch.meshgrid(uv, uv, indexing='ij')
    pixels = torch.stack((x, y, torch.ones_like(x)), -1)
    c2w = torch.linalg.inv(extrinsics.float())
    rays = pixels @ torch.linalg.inv(intrinsics.float()).T @ c2w[:3, :3].T
    origin = c2w[:3, 3]
    # Intersect each ray with the unit world cube; depths are camera Z values.
    safe_rays = torch.where(rays.abs() < 1e-7, torch.full_like(rays, 1e-7), rays)
    a, b = (-0.5-origin)/safe_rays, (0.5-origin)/safe_rays
    near = torch.minimum(a, b).amax(-1).clamp_min(0)
    far = torch.maximum(a, b).amin(-1)
    hit = far > near
    frac = (torch.arange(samples, device=device, dtype=torch.float32)+0.5)/samples
    depth = near[None] + frac[:, None, None] * (far-near)[None].clamp_min(0)
    points = origin + depth[..., None] * rays[None]
    # grid_sample indexes W,H,D; our volume axes are x,y,z.
    grid = (points[..., [2, 1, 0]] * 2).unsqueeze(0)
    density = F.grid_sample(volume.float(), grid, align_corners=False,
                            padding_mode='zeros')[0, 0]
    length = (far-near).clamp_min(0) * rays.norm(dim=-1) / samples
    alpha = -torch.expm1(-(density * length[None] * volume.shape[-1]).sum(0))
    return alpha * hit


class SSRenderLossMixin:
    def _init_ss_render(self, render_loss, render_loss_decoder):
        self.ss_render_cfg = RenderLossConfig.from_dict(render_loss)
        self.ss_render_cfg.sigma_min = self.sigma_min
        self.ss_render_decoder_config = render_loss_decoder or {}
        self.ss_render_decoder = None

    def _ss_render_loss(self, x_t, pred, t, x_0, case_dirs, edit_type):
        cfg = self.ss_render_cfg
        zero = pred.new_zeros((), dtype=torch.float32)
        result = dict(render_loss=zero, render_loss_mse=zero, render_n_used=0,
                      render_lambda=0.0, render_loss_weighted=zero)
        step = self.step + 1
        lam = schedule_lambda_render(step, self.max_steps, cfg)
        if not cfg.enabled or lam <= 0 or step % max(cfg.every_n_steps, 1):
            return zero, result
        eligible = [i for i in range(len(t)) if t[i].item() < cfg.t_max and
                    (cfg.allowed_edit_types is None or edit_type[i] in cfg.allowed_edit_types)]
        if cfg.first_only:
            eligible = eligible[:1]
        if not eligible:
            return zero, result
        if case_dirs is None:
            raise ValueError('SS render loss requires case_dirs with camera metadata')
        if self.ss_render_decoder is None:
            from ...models import from_pretrained
            path = self.ss_render_decoder_config.get('pretrained',
                'microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16')
            self.ss_render_decoder = from_pretrained(path, use_fp16=False).to(pred.device).eval()
            self.ss_render_decoder.requires_grad_(False)
        tt = t[:, None, None, None, None]
        clean = predict_clean_features(x_t.float(), pred.float(), tt, self.sigma_min)
        losses = []
        with torch.amp.autocast('cuda', enabled=False):
            for i in eligible:
                logits = self.ss_render_decoder(clean[i:i+1])
                with torch.no_grad():
                    gt = (self.ss_render_decoder(x_0[i:i+1].float()) > 0).float()
                ext, intr = load_render_camera(case_dirs[i], cfg)
                ext, intr = ext.to(pred.device), intr.to(pred.device)
                # The pretrained occupancy logits reach |logit| > 100. A
                # temperature keeps the training silhouette differentiable.
                image = render_occupancy((logits/cfg.occupancy_temperature).sigmoid(),
                                         ext, intr, cfg.resolution, cfg.ray_samples)
                target = render_occupancy(gt, ext, intr, cfg.resolution, cfg.ray_samples)
                losses.append(F.mse_loss(image, target))
        loss = torch.stack(losses).mean()
        result.update(render_loss=loss.detach(), render_loss_mse=loss.detach(),
                      render_n_used=len(eligible), render_lambda=lam,
                      render_loss_weighted=(lam*loss).detach())
        return lam*loss, result
