from omegaconf import OmegaConf

from .diffglue.base_model import BaseModel

to_ctr = OmegaConf.to_container  # convert DictConfig to dict


import importlib.util

from .diffglue.base_model import BaseModel
from .diffglue.diffuser import SpacedDiffusion
from .diffglue.diffglue import DiffGlue


def get_class(mod_path, BaseClass):
    """Get the class object which inherits from BaseClass and is defined in
    the module named mod_name, child of base_path.
    """
    import inspect

    mod = __import__(mod_path, fromlist=[""])
    classes = inspect.getmembers(mod, inspect.isclass)
    # Filter classes defined in the module
    classes = [c for c in classes if c[1].__module__ == mod_path]
    # Filter classes inherited from BaseModel
    classes = [c for c in classes if issubclass(c[1], BaseClass)]
    assert len(classes) == 1, classes
    return classes[0][1]


def get_model(name):
    import_paths = [
        "." + name,
    ]
    for path in import_paths:
        try:
            spec = importlib.util.find_spec(path)
        except ModuleNotFoundError:
            spec = None
        if spec is not None:
            try:
                return get_class(path, BaseModel)
            except AssertionError:
                mod = __import__(path, fromlist=[""])
                try:
                    return mod.__main_model__
                except AttributeError as exc:
                    print(exc)
                    continue

    raise RuntimeError(f'Model {name} not found in any of [{" ".join(import_paths)}]')


class DiffGluePipeline(BaseModel):
    default_conf = {
        "diffuser": {
            "name": "diffglue.diffuser",
            "steps": 4096,
            "learn_sigma": False,
            "sigma_small": False,
            "noise_schedule": "linear",
            "use_kl": False,
            "predict_xstart": True,
            "rescale_timesteps": True,
            "rescale_learned_sigmas": True,
            "timestep_respacing": "",
            "ddim_steps": 2,
            "schedule_sampler": "uniform",
            "use_ddim": True,
            "clip_denoised": True,
            "diffuser_loss_weight": 1,
            "scale": 2,
        },
        "matcher": {
            "name": "diffglue.diffglue",
            "features": "superpoint",
            "depth_confidence": -1,
            "width_confidence": -1,
            "filter_threshold": 0.1,
            "flash": False,
            "checkpointed": True,
            "n_layers": 9,
            "scale": 2,
        },
    }
    required_data_keys = ["image0", "image1"]
    strict_conf = False  # need to pass new confs to children models
    components = [
        "diffuser",
        "matcher",
    ]

    def _init(self, conf):
        if conf.diffuser.name:
            self.eval_diffuser = SpacedDiffusion({**to_ctr(conf.diffuser), **{"timestep_respacing": str(conf.diffuser.ddim_steps)}})

        if conf.matcher.name:
            self.matcher = DiffGlue(to_ctr(conf.matcher))

    def _forward(self, data):
        pred = {}
        if self.conf.diffuser.name and self.conf.matcher.name:
            pred = {**pred, **self.eval_diffuser(self.matcher, {**data, **pred})}
        return pred