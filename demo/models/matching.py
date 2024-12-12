import torch

from .superpoint import SuperPoint
from .diffglue_pipeline import DiffGluePipeline

from pathlib import Path
from omegaconf import OmegaConf


def computeNN(desc_ii, desc_jj):
    desc_ii, desc_jj = desc_ii.squeeze(0).transpose(0,1), desc_jj.squeeze(0).transpose(0,1)
    d1 = (desc_ii**2).sum(1)
    d2 = (desc_jj**2).sum(1)
    distmat = (d1.unsqueeze(1) + d2.unsqueeze(0) - 2*torch.matmul(desc_ii, desc_jj.transpose(0,1))).sqrt()
    distVals, nnIdx1 = torch.topk(distmat, k=2, dim=1, largest=False)
    nnIdx1 = nnIdx1[:,0]
    _, nnIdx2 = torch.topk(distmat, k=1, dim=0, largest=False)
    nnIdx2= nnIdx2.squeeze()
    mutual_nearest = nnIdx2[nnIdx1] == torch.arange(nnIdx1.shape[0]).cuda()
    ratio_test = distVals[:,0] / distVals[:,1].clamp(min=1e-10)
    idx_sort = [torch.arange(nnIdx1.shape[0]), nnIdx1]
    return idx_sort, ratio_test, mutual_nearest


class Matching(torch.nn.Module):
    """ Image Matching Frontend (SuperPoint + DiffGlue) """
    def __init__(self, config={}):
        super().__init__()
        self.superpoint = SuperPoint(config.get('superpoint', {}))

        default_conf = OmegaConf.create(DiffGluePipeline.default_conf)
        self.diffglue = DiffGluePipeline(default_conf).eval().cuda()  # load the matcher

        print('Loaded DiffGlue model')
        exper = Path("./models/weights/SP_DiffGlue.tar")
        ckpt = exper
        ckpt = torch.load(str(ckpt), map_location="cpu")

        state_dict = ckpt["model"]
        dict_params = set(state_dict.keys())
        model_params = set(map(lambda n: n[0], self.diffglue.named_parameters()))
        diff = model_params - dict_params
        if len(diff) > 0:
            state_dict = {k.replace('matcher.', 'matcher.net.'): v for k, v in state_dict.items()}
        self.diffglue.load_state_dict(state_dict, strict=False)


    def forward(self, data):
        """ Run SuperPoint and DiffGlue """
        pred = {}

        # Extract SuperPoint (keypoints, scores, descriptors) if not provided
        if 'keypoints0' not in data:
            pred0 = self.superpoint({'image': data['image0']})
            pred = {**pred, **{k+'0': v for k, v in pred0.items()}}
        if 'keypoints1' not in data:
            pred1 = self.superpoint({'image': data['image1']})
            pred = {**pred, **{k+'1': v for k, v in pred1.items()}}

        # Batch all features
        data = {**data, **pred}

        for k in data:
            if isinstance(data[k], (list, tuple)):
                data[k] = torch.stack(data[k])

        # Perform the matching
        pred = {**self.diffglue(data)}
        pred = {**data, **pred}

        return pred
