from typing import Any
import numpy as np
import torch as th

from omegaconf import OmegaConf
from . import gaussian_diffusion as gd
from .gaussian_diffusion import GaussianDiffusion
from .resample import create_named_schedule_sampler


def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.

    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.

    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.

    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


class SpacedDiffusion(GaussianDiffusion):
    """
    A diffusion process which can skip steps in a base diffusion process.

    :param use_timesteps: a collection (sequence or set) of timesteps from the
                          original diffusion process to retain.
    :param kwargs: the kwargs to create the base diffusion process.
    """
    default_conf = {
        "steps": 1000,
        "learn_sigma": False,
        "sigma_small": False,
        "noise_schedule": "linear",
        "use_kl": False,
        "predict_xstart": False,
        "rescale_timesteps": False,
        "rescale_learned_sigmas": False,
        "timestep_respacing": "",
        "schedule_sampler": "uniform",
        "use_ddim": False,
        "clip_denoised": False,
        "diffuser_loss_weight": 1000,
        "scale": 1,
    }

    def __init__(self, conf):
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        if not conf.timestep_respacing:
            self.conf.timestep_respacing = [conf.steps]
            conf.timestep_respacing = [conf.steps]
        use_timesteps=space_timesteps(conf.steps, conf.timestep_respacing)
        if conf.use_kl:
            loss_type = gd.LossType.RESCALED_KL
        elif conf.rescale_learned_sigmas:
            loss_type = gd.LossType.RESCALED_MSE
        else:
            loss_type = gd.LossType.MSE
        betas=gd.get_named_beta_schedule(conf.noise_schedule, conf.steps)
        model_mean_type=(
            gd.ModelMeanType.EPSILON if not conf.predict_xstart else gd.ModelMeanType.START_X
        )
        model_var_type=(
            (
                gd.ModelVarType.FIXED_LARGE
                if not conf.sigma_small
                else gd.ModelVarType.FIXED_SMALL
            )
            if not conf.learn_sigma
            else gd.ModelVarType.LEARNED_RANGE
        )
        loss_type=loss_type
        rescale_timesteps=conf.rescale_timesteps

        self.use_timesteps = set(use_timesteps)
        self.timestep_map = []
        self.original_num_steps = len(betas)

        kwargs = {
            "betas": betas,
            "model_mean_type": model_mean_type,
            "model_var_type": model_var_type,
            "loss_type": loss_type,
            "rescale_timesteps": rescale_timesteps,
            "scale": conf.scale,
        }

        base_diffusion = GaussianDiffusion(**kwargs)  # pylint: disable=missing-kwoa
        last_alpha_cumprod = 1.0
        new_betas = []
        for i, alpha_cumprod in enumerate(base_diffusion.alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
                self.timestep_map.append(i)
        kwargs["betas"] = np.array(new_betas)
        super().__init__(**kwargs)

    def p_mean_variance(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)

    def training_losses(
        self, model, *args, **kwargs
    ):  # pylint: disable=signature-differs
        return super().training_losses(self._wrap_model(model), *args, **kwargs)

    def sample(self, model, cond=None):
        if cond is None:
            cond = {}
        sample_fn = (
            self.p_sample_loop if not self.conf.use_ddim else self.ddim_sample_loop
        )
        sample = sample_fn(
            model,
            (cond["data"]["keypoints0"].shape[0], 1, cond["data"]["keypoints0"].shape[1]+1, cond["data"]["keypoints1"].shape[1]+1), # cond["keypoints0"]: BNC
            clip_denoised=self.conf.clip_denoised,
            model_kwargs=cond,
        ) if not self.conf.use_ddim else sample_fn(
            model,
            (cond["data"]["keypoints0"].shape[0], 1, cond["data"]["keypoints0"].shape[1]+1, cond["data"]["keypoints1"].shape[1]+1), # cond["keypoints0"]: BNC
            clip_denoised=self.conf.clip_denoised,
            # ddim_steps=self.ddim_steps,
            model_kwargs=cond,
        )
        sample["sample"] = sample["sample"].contiguous()
        return sample

    def _wrap_model(self, model):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(
            model, self.timestep_map, self.rescale_timesteps, self.original_num_steps
        )

    def _scale_timesteps(self, t):
        # Scaling is done by the wrapped model.
        return t
    
    def __call__(
        self, model, pred
    ):  # pylint: disable=signature-differs
        if model.training:
            self.training = True
            self.schedule_sampler = create_named_schedule_sampler(self.conf.schedule_sampler, self)
            m, n = pred["gt_matches0"].size(-1), pred["gt_matches1"].size(-1)
            positive = pred["gt_assignment"].float()
            neg0 = (pred["gt_matches0"] == -1).float()
            neg1 = (pred["gt_matches1"] == -1).float()
            x_start = th.zeros(pred["gt_assignment"].shape[0], 1, m+1, n+1).to(pred["gt_assignment"].device)
            x_start[:, 0, :-1, :-1] = positive
            x_start[:, 0, :-1, -1] = neg0
            x_start[:, 0, -1, :-1] = neg1
            x_start[..., :-1, :-1] = (x_start[..., :-1, :-1]-0.5)*self.conf.scale
            x_start[..., :-1,  -1] = (x_start[..., :-1,  -1]-0.5)*self.conf.scale
            x_start[...,  -1, :-1] = (x_start[...,  -1, :-1]-0.5)*self.conf.scale
            t, weights = self.schedule_sampler.sample(pred["gt_assignment"].shape[0], pred["gt_assignment"].device)
            args = (x_start, t)
            kwargs = {
                "model_kwargs": {"data": pred},
                "noise": None,
            }
            results = self.training_losses(model, *args, **kwargs) # "loss", "vb"
            results["diffuser_loss"] = results["diffuser_loss"]*weights
            return results
        else:
            self.training = False
            return self.sample(model, cond={"data": pred})

    def loss(self, pred, data):
        if self.training:
            diffuser_loss = pred["diffuser_loss"]
            losses = {"diffuser_total": diffuser_loss}
            metrics = {}
            return losses, metrics
        else:
            raise NotImplementedError

class _WrappedModel:
    def __init__(self, model, timestep_map, rescale_timesteps, original_num_steps):
        self.model = model
        self.timestep_map = timestep_map
        self.rescale_timesteps = rescale_timesteps
        self.original_num_steps = original_num_steps

    def __call__(self, x, ts, **kwargs):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_steps)
        return self.model(x, new_ts, **kwargs)

__main_model__ = SpacedDiffusion