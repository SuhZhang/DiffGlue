import warnings
from typing import Callable, List, Optional

import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import nn
from torch.utils.checkpoint import checkpoint


def weight_loss(log_assignment, weights, gamma=0.0):
    b, m, n = log_assignment.shape
    m -= 1
    n -= 1

    loss_sc = log_assignment * weights

    num_neg0 = weights[:, :m, -1].sum(-1).clamp(min=1.0)
    num_neg1 = weights[:, -1, :n].sum(-1).clamp(min=1.0)
    num_pos = weights[:, :m, :n].sum((-1, -2)).clamp(min=1.0)

    nll_pos = -loss_sc[:, :m, :n].sum((-1, -2))
    nll_pos /= num_pos.clamp(min=1.0)

    nll_neg0 = -loss_sc[:, :m, -1].sum(-1)
    nll_neg1 = -loss_sc[:, -1, :n].sum(-1)

    nll_neg = (nll_neg0 + nll_neg1) / (num_neg0 + num_neg1)

    return nll_pos, nll_neg, num_pos, (num_neg0 + num_neg1) / 2.0

class NLLLoss(nn.Module):
    default_conf = {
        "nll_balancing": 0.5,
        "gamma_f": 0.0,  # focal loss
    }

    def __init__(self, conf):
        super().__init__()
        self.conf = OmegaConf.merge(self.default_conf, conf)
        self.loss_fn = self.nll_loss

    def forward(self, pred, data, weights=None):
        log_assignment = pred["log_assignment"]
        if weights is None:
            weights = self.loss_fn(log_assignment, data)
        nll_pos, nll_neg, num_pos, num_neg = weight_loss(
            log_assignment, weights, gamma=self.conf.gamma_f
        )
        nll = (
            self.conf.nll_balancing * nll_pos + (1 - self.conf.nll_balancing) * nll_neg
        )

        return (
            nll,
            weights,
            {
                "assignment_nll": nll,
                "nll_pos": nll_pos,
                "nll_neg": nll_neg,
                "num_matchable": num_pos,
                "num_unmatchable": num_neg,
            },
        )

    def nll_loss(self, log_assignment, data):
        m, n = data["gt_matches0"].size(-1), data["gt_matches1"].size(-1)
        positive = data["gt_assignment"].float()
        neg0 = (data["gt_matches0"] == -1).float()
        neg1 = (data["gt_matches1"] == -1).float()

        weights = torch.zeros_like(log_assignment)
        weights[:, :m, :n] = positive

        weights[:, :m, -1] = neg0
        weights[:, -1, :n] = neg1
        return weights
    
@torch.no_grad()
def matcher_metrics(pred, data, prefix="", prefix_gt=None):
    def recall(m, gt_m):
        mask = (gt_m > -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def accuracy(m, gt_m):
        mask = (gt_m >= -1).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def precision(m, gt_m):
        mask = ((m > -1) & (gt_m >= -1)).float()
        return ((m == gt_m) * mask).sum(1) / (1e-8 + mask.sum(1))

    def ranking_ap(m, gt_m, scores):
        p_mask = ((m > -1) & (gt_m >= -1)).float()
        r_mask = (gt_m > -1).float()
        sort_ind = torch.argsort(-scores)
        sorted_p_mask = torch.gather(p_mask, -1, sort_ind)
        sorted_r_mask = torch.gather(r_mask, -1, sort_ind)
        sorted_tp = torch.gather(m == gt_m, -1, sort_ind)
        p_pts = torch.cumsum(sorted_tp * sorted_p_mask, -1) / (
            1e-8 + torch.cumsum(sorted_p_mask, -1)
        )
        r_pts = torch.cumsum(sorted_tp * sorted_r_mask, -1) / (
            1e-8 + sorted_r_mask.sum(-1)[:, None]
        )
        r_pts_diff = r_pts[..., 1:] - r_pts[..., :-1]
        return torch.sum(r_pts_diff * p_pts[:, None, -1], dim=-1)

    if prefix_gt is None:
        prefix_gt = prefix
    rec = recall(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    prec = precision(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    acc = accuracy(pred[f"{prefix}matches0"], data[f"gt_{prefix_gt}matches0"])
    ap = ranking_ap(
        pred[f"{prefix}matches0"],
        data[f"gt_{prefix_gt}matches0"],
        pred[f"{prefix}matching_scores0"],
    )
    metrics = {
        f"{prefix}match_recall": rec,
        f"{prefix}match_precision": prec,
        f"{prefix}accuracy": acc,
        f"{prefix}average_precision": ap,
    }
    return metrics

import math

def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


FLASH_AVAILABLE = hasattr(F, "scaled_dot_product_attention")

torch.backends.cudnn.deterministic = True


@torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
def normalize_keypoints(
    kpts: torch.Tensor, size: Optional[torch.Tensor] = None
) -> torch.Tensor:
    if size is None:
        size = 1 + kpts.max(-2).values - kpts.min(-2).values
    elif not isinstance(size, torch.Tensor):
        size = torch.tensor(size, device=kpts.device, dtype=kpts.dtype)
    size = size.to(kpts)
    shift = size / 2
    scale = size.max(-1).values / 2
    kpts = (kpts - shift[..., None, :]) / scale[..., None, None]
    return kpts


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.unflatten(-1, (-1, 2))
    x1, x2 = x.unbind(dim=-1)
    return torch.stack((-x2, x1), dim=-1).flatten(start_dim=-2)


def apply_cached_rotary_emb(freqs: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return (t * freqs[0]) + (rotate_half(t) * freqs[1])


class LearnableFourierPositionalEncoding(nn.Module):
    def __init__(self, M: int, dim: int, F_dim: int = None, gamma: float = 1.0) -> None:
        super().__init__()
        F_dim = F_dim if F_dim is not None else dim
        self.gamma = gamma
        self.Wr = nn.Linear(M, F_dim // 2, bias=False)
        nn.init.normal_(self.Wr.weight.data, mean=0, std=self.gamma**-2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """encode position vector"""
        projected = self.Wr(x)
        cosines, sines = torch.cos(projected), torch.sin(projected)
        emb = torch.stack([cosines, sines], 0).unsqueeze(-3)
        return emb.repeat_interleave(2, dim=-1)


class TokenConfidence(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.token = nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid())
        self.loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """get confidence tokens"""
        return (
            self.token(desc0.detach()).squeeze(-1),
            self.token(desc1.detach()).squeeze(-1),
        )

    def loss(self, desc0, desc1, la_now, la_final):
        logit0 = self.token[0](desc0.detach()).squeeze(-1)
        logit1 = self.token[0](desc1.detach()).squeeze(-1)
        la_now, la_final = la_now.detach(), la_final.detach()
        correct0 = (
            la_final[:, :-1, :].max(-1).indices == la_now[:, :-1, :].max(-1).indices
        )
        correct1 = (
            la_final[:, :, :-1].max(-2).indices == la_now[:, :, :-1].max(-2).indices
        )
        return (
            self.loss_fn(logit0, correct0.float()).mean(-1)
            + self.loss_fn(logit1, correct1.float()).mean(-1)
        ) / 2.0


class Attention(nn.Module):
    def __init__(self, allow_flash: bool) -> None:
        super().__init__()
        if allow_flash and not FLASH_AVAILABLE:
            warnings.warn(
                "FlashAttention is not available. For optimal speed, "
                "consider installing torch >= 2.0 or flash-attn.",
                stacklevel=2,
            )
        self.enable_flash = allow_flash and FLASH_AVAILABLE

        if FLASH_AVAILABLE:
            torch.backends.cuda.enable_flash_sdp(allow_flash)

    def forward(self, q, k, v, adj_mat: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.enable_flash and q.device.type == "cuda":
            # use torch 2.0 scaled_dot_product_attention with flash
            if FLASH_AVAILABLE:
                if adj_mat is not None:
                    v_add = torch.einsum("...ij,...jd->...id", F.softmax(adj_mat, -1).unsqueeze(1), v)
                args = [x.half().contiguous() for x in [q, k, v]]
                v = F.scaled_dot_product_attention(*args, attn_mask=mask).to(q.dtype)
                if adj_mat is not None:
                    v += v_add
                return v if mask is None else v.nan_to_num()
        elif FLASH_AVAILABLE:
            if adj_mat is not None:
                v_add = torch.einsum("...ij,...jd->...id", F.softmax(adj_mat, -1).unsqueeze(1), v)
            args = [x.contiguous() for x in [q, k, v]]
            v = F.scaled_dot_product_attention(*args, attn_mask=mask)
            if adj_mat is not None:
                v += v_add
            return v if mask is None else v.nan_to_num()
        else:
            s = q.shape[-1] ** -0.5
            sim = torch.einsum("...id,...jd->...ij", q, k) * s
            if mask is not None:
                sim.masked_fill(~mask, -float("inf"))
            if adj_mat is not None:
                attn = F.softmax(sim, -1) + F.softmax(adj_mat, -1).unsqueeze(1)
            else:
                attn = F.softmax(sim, -1)
            return torch.einsum("...ij,...jd->...id", attn, v)


class SelfBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        assert self.embed_dim % num_heads == 0
        self.head_dim = self.embed_dim // num_heads
        self.Wqkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.inner_attn = Attention(flash)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def forward(
        self,
        x: torch.Tensor,
        encoding: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        qkv = self.Wqkv(x)
        qkv = qkv.unflatten(-1, (self.num_heads, -1, 3)).transpose(1, 2)
        q, k, v = qkv[..., 0], qkv[..., 1], qkv[..., 2]
        q = apply_cached_rotary_emb(encoding, q)
        k = apply_cached_rotary_emb(encoding, k)
        context = self.inner_attn(q, k, v, mask=mask)
        message = self.out_proj(context.transpose(1, 2).flatten(start_dim=-2))
        return x + self.ffn(torch.cat([x, message], -1))


class CrossBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int, flash: bool = False, bias: bool = True
    ) -> None:
        super().__init__()
        self.heads = num_heads
        dim_head = embed_dim // num_heads
        self.scale = dim_head**-0.5
        inner_dim = dim_head * num_heads
        self.to_qk = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )
        if flash and FLASH_AVAILABLE:
            self.flash = Attention(True)
        else:
            self.flash = None

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)

    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, encoding0: torch.Tensor = None, encoding1: torch.Tensor = None, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        qk0, qk1 = self.map_(self.to_qk, x0, x1)
        v0, v1 = self.map_(self.to_v, x0, x1)
        qk0, qk1, v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (qk0, qk1, v0, v1),
        )
        if encoding0 is not None and encoding1 is not None:
            qk0 = apply_cached_rotary_emb(encoding0, qk0)
            qk1 = apply_cached_rotary_emb(encoding1, qk1)
        if self.flash is not None and qk0.device.type == "cuda":
            m0 = self.flash(qk0, qk1, v1, mask=mask)
            m1 = self.flash(
                qk1, qk0, v0, mask=mask.transpose(-1, -2) if mask is not None else None
            )
        else:
            qk0, qk1 = qk0 * self.scale**0.5, qk1 * self.scale**0.5
            sim = torch.einsum("bhid, bhjd -> bhij", qk0, qk1)
            if mask is not None:
                sim = sim.masked_fill(~mask, -float("inf"))
            attn01 = F.softmax(sim, dim=-1)
            attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
            m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
            m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
            if mask is not None:
                m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class AdjBlock(nn.Module):
    def __init__(
        self, embed_dim: int, num_heads: int = 1, bias: bool = True #, slope: float = 0.1
    ) -> None:
        super().__init__()
        self.heads = num_heads
        # self.slope = slope
        dim_head = embed_dim // num_heads
        inner_dim = dim_head * num_heads
        self.to_v = nn.Linear(embed_dim, inner_dim, bias=bias)
        self.to_out = nn.Linear(inner_dim, embed_dim, bias=bias)
        self.ffn = nn.Sequential(
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.LayerNorm(2 * embed_dim, elementwise_affine=True),
            nn.GELU(),
            nn.Linear(2 * embed_dim, embed_dim),
        )

    def map_(self, func: Callable, x0: torch.Tensor, x1: torch.Tensor):
        return func(x0), func(x1)
    
    def forward(
        self, x0: torch.Tensor, x1: torch.Tensor, adj_mat: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        v0, v1 = self.map_(self.to_v, x0, x1)
        v0, v1 = map(
            lambda t: t.unflatten(-1, (self.heads, -1)).transpose(1, 2),
            (v0, v1),
        )
        sim = adj_mat.unsqueeze(1)
        if mask is not None:
            sim = sim.masked_fill(~mask, -float("inf"))
        attn01 = F.softmax(sim, dim=-1)
        attn10 = F.softmax(sim.transpose(-2, -1).contiguous(), dim=-1)
        m0 = torch.einsum("bhij, bhjd -> bhid", attn01, v1)
        m1 = torch.einsum("bhji, bhjd -> bhid", attn10.transpose(-2, -1), v0)
        if mask is not None:
            m0, m1 = m0.nan_to_num(), m1.nan_to_num()
        m0, m1 = self.map_(lambda t: t.transpose(1, 2).flatten(start_dim=-2), m0, m1)
        m0, m1 = self.map_(self.to_out, m0, m1)
        x0 = x0 + self.ffn(torch.cat([x0, m0], -1))
        x1 = x1 + self.ffn(torch.cat([x1, m1], -1))
        return x0, x1


class TransformerLayer(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.atten_layers = 1
        self.embd_layers = nn.ModuleList(
            [nn.Sequential(
                nn.ReLU(),
                nn.Linear(args[2], args[0]),
            ) for _ in range(self.atten_layers)]
        )
        self_attn_args = (args[0], args[1], args[-1])
        self_attn_kwargs = {}
        self.self_attn = nn.ModuleList(
            [SelfBlock(*self_attn_args, **self_attn_kwargs) for _ in range(self.atten_layers)]
        )
        adj_attn_args = (args[0],)
        adj_attn_kwargs = {}
        self.adj_attn = nn.ModuleList(
            [AdjBlock(*adj_attn_args, **adj_attn_kwargs) for _ in range(self.atten_layers)]
        )
        cross_attn_args = (args[0], args[1], args[-1])
        cross_attn_kwargs = {}
        self.cross_attn = nn.ModuleList(
            [CrossBlock(*cross_attn_args, **cross_attn_kwargs) for _ in range(self.atten_layers)]
        )
        self.desc_compact = nn.ModuleList(
            [nn.Sequential(
                nn.Linear(2*args[0], args[0]),
            ) for _ in range(self.atten_layers)]
        )

    def forward(
        self,
        desc0,
        desc1,
        encoding0,
        encoding1,
        embd,
        adj_mat: Optional[torch.Tensor] = None,
        mask0: Optional[torch.Tensor] = None,
        mask1: Optional[torch.Tensor] = None,
    ):
        if mask0 is not None and mask1 is not None:
            return self.masked_forward(desc0, desc1, encoding0, encoding1, embd, adj_mat, mask0, mask1)
        else:
            for i in range(self.atten_layers):
                desc0 = self.self_attn[i](desc0, encoding0)
                desc1 = self.self_attn[i](desc1, encoding1)
                cross_embd = self.embd_layers[i](embd).type(desc0.dtype)
                desc0_adj = desc0 + cross_embd.unsqueeze(1)
                desc1_adj = desc1 + cross_embd.unsqueeze(1)
                desc0_adj, desc1_adj = self.adj_attn[i](desc0_adj, desc1_adj, adj_mat)
                desc0, desc1 = self.desc_compact[i](torch.cat([desc0, desc0_adj], dim=-1)), self.desc_compact[i](torch.cat([desc1, desc1_adj], dim=-1))
                desc0, desc1 = self.cross_attn[i](desc0, desc1)
            return desc0, desc1

    # This part is compiled and allows padding inputs
    def masked_forward(self, desc0, desc1, encoding0, encoding1, embd, adj_mat, mask0, mask1):
        mask = mask0 & mask1.transpose(-1, -2)
        mask0 = mask0 & mask0.transpose(-1, -2)
        mask1 = mask1 & mask1.transpose(-1, -2)
        for i in range(self.atten_layers):
            desc0 = self.self_attn[i](desc0, encoding0)
            desc1 = self.self_attn[i](desc1, encoding1)
            cross_embd = self.embd_layers[i](embd).type(desc0.dtype)
            desc0_adj = desc0 + cross_embd.unsqueeze(1)
            desc1_adj = desc1 + cross_embd.unsqueeze(1)
            desc0_adj, desc1_adj = self.adj_attn[i](desc0_adj, desc1_adj, adj_mat, mask)
            desc0, desc1 = self.desc_compact[i](torch.cat([desc0, desc0_adj], dim=-1)), self.desc_compact[i](torch.cat([desc1, desc1_adj], dim=-1))
            desc0, desc1 = self.cross_attn[i](desc0, desc1, mask)
        return desc0, desc1


def sigmoid_log_double_softmax(
    sim: torch.Tensor, z0: torch.Tensor, z1: torch.Tensor
) -> torch.Tensor:
    """create the log assignment matrix from logits and similarity"""
    b, m, n = sim.shape
    certainties = F.logsigmoid(z0) + F.logsigmoid(z1).transpose(1, 2)
    scores0 = F.log_softmax(sim, 2)
    scores1 = F.log_softmax(sim.transpose(-1, -2).contiguous(), 2).transpose(-1, -2)
    scores = sim.new_full((b, m + 1, n + 1), 0)
    scores[:, :m, :n] = scores0 + scores1 + certainties
    scores[:, :-1, -1] = F.logsigmoid(-z0.squeeze(-1))
    scores[:, -1, :-1] = F.logsigmoid(-z1.squeeze(-1))
    return scores


class MatchAssignment(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.matchability = nn.Linear(dim, 1, bias=True)
        self.final_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, desc0: torch.Tensor, desc1: torch.Tensor):
        """build assignment matrix from descriptors"""
        mdesc0, mdesc1 = self.final_proj(desc0), self.final_proj(desc1)
        _, _, d = mdesc0.shape
        mdesc0, mdesc1 = mdesc0 / d**0.25, mdesc1 / d**0.25
        sim = torch.einsum("bmd,bnd->bmn", mdesc0, mdesc1)
        z0 = self.matchability(desc0)
        z1 = self.matchability(desc1)
        scores = sigmoid_log_double_softmax(sim, z0, z1)
        return scores, sim

    def get_matchability(self, desc: torch.Tensor):
        return torch.sigmoid(self.matchability(desc)).squeeze(-1)


def filter_matches(scores: torch.Tensor, th: float):
    """obtain matches from a log assignment matrix [Bx M+1 x N+1]"""
    max0, max1 = scores[:, :-1, :-1].max(2), scores[:, :-1, :-1].max(1)
    m0, m1 = max0.indices, max1.indices
    indices0 = torch.arange(m0.shape[1], device=m0.device)[None]
    indices1 = torch.arange(m1.shape[1], device=m1.device)[None]
    mutual0 = indices0 == m1.gather(1, m0)
    mutual1 = indices1 == m0.gather(1, m1)
    max0_exp = max0.values.exp()
    zero = max0_exp.new_tensor(0)
    mscores0 = torch.where(mutual0, max0_exp, zero)
    mscores1 = torch.where(mutual1, mscores0.gather(1, m1), zero)
    valid0 = mutual0 & (mscores0 > th)
    valid1 = mutual1 & valid0.gather(1, m1)
    m0 = torch.where(valid0, m0, -1)
    m1 = torch.where(valid1, m1, -1)
    return m0, m1, mscores0, mscores1


class DiffGlue(nn.Module):
    default_conf = {
        "name": "diffglue",  # just for interfacing
        "input_dim": 256,  # input descriptor dimension (autoselected from weights)
        "add_scale_ori": False,
        "descriptor_dim": 256,
        "n_layers": -1,
        "num_heads": 4,
        "flash": False,  # enable FlashAttention if available.
        "mp": False,  # enable mixed precision
        "filter_threshold": 0.0,  # match threshold
        "checkpointed": False,
        "loss": {
            "gamma": 1.0,
            "fn": "nll",
            "nll_balancing": 0.5,
        },
    }

    required_data_keys = ["keypoints0", "keypoints1", "descriptors0", "descriptors1"]

    def __init__(self, conf) -> None:
        super().__init__()
        self.conf = conf = OmegaConf.merge(self.default_conf, conf)

        self.time_embed_channels = conf.descriptor_dim * 4
        self.time_embed = nn.Sequential(
            nn.Linear(conf.descriptor_dim, self.time_embed_channels), 
            nn.ReLU(), 
            nn.Linear(self.time_embed_channels, self.time_embed_channels), 
        )

        if conf.input_dim != conf.descriptor_dim:
            self.input_proj = nn.Linear(conf.input_dim, conf.descriptor_dim, bias=True)
        else:
            self.input_proj = nn.Identity()

        head_dim = conf.descriptor_dim // conf.num_heads
        self.posenc = LearnableFourierPositionalEncoding(
            2 + 2 * conf.add_scale_ori, head_dim, head_dim
        )

        h, n, d = conf.num_heads, conf.n_layers, conf.descriptor_dim

        self.transformers = nn.ModuleList(
            [TransformerLayer(d, h, self.time_embed_channels, conf.flash) for layer_index in range(n)]
        )

        self.log_assignment = nn.ModuleList([MatchAssignment(d) for _ in range(n)])
        self.token_confidence = nn.ModuleList(
            # [TokenConfidence(d) for _ in range(n - 1)]
            [TokenConfidence(d) for _ in range(n)]
        )

        self.loss_fn = NLLLoss(conf.loss)


    def compile(self, mode="reduce-overhead"):
        if self.conf.width_confidence != -1:
            warnings.warn(
                "Point pruning is partially disabled for compiled forward.",
                stacklevel=2,
            )

        for i in range(self.conf.n_layers):
            self.transformers[i] = torch.compile(
                self.transformers[i], mode=mode, fullgraph=True
            )

    def forward(self, adj_mat_fore, timesteps, data: dict) -> dict:
        adj_mat_fore[...,:-1,:-1] = adj_mat_fore[...,:-1,:-1]/self.conf.scale+0.5
        adj_mat_fore[...,:-1,-1] = adj_mat_fore[...,:-1,-1]/self.conf.scale+0.5
        adj_mat_fore[...,-1,:-1] = adj_mat_fore[...,-1,:-1]/self.conf.scale+0.5
        adj_mat_fore = adj_mat_fore.squeeze(1)
        time_embd = self.time_embed(timestep_embedding(timesteps, self.conf.descriptor_dim))

        for key in self.required_data_keys:
            assert key in data, f"Missing key {key} in data"
        data0, data1 = data["image0"], data["image1"]
        kpts0, kpts1 = data["keypoints0"], data["keypoints1"]
        b, m, _ = kpts0.shape
        b, n, _ = kpts1.shape
        device = kpts0.device
        size0, size1 = data0.shape[2:], data1.shape[2:]
        kpts0 = normalize_keypoints(kpts0, size0).clone()
        kpts1 = normalize_keypoints(kpts1, size1).clone()

        if self.conf.add_scale_ori:
            sc0, o0 = data["scales0"], data["oris0"]
            sc1, o1 = data["scales1"], data["oris1"]
            kpts0 = torch.cat(
                [
                    kpts0,
                    sc0 if sc0.dim() == 3 else sc0[..., None],
                    o0 if o0.dim() == 3 else o0[..., None],
                ],
                -1,
            )
            kpts1 = torch.cat(
                [
                    kpts1,
                    sc1 if sc1.dim() == 3 else sc1[..., None],
                    o1 if o1.dim() == 3 else o1[..., None],
                ],
                -1,
            )

        desc0 = data["descriptors0"].detach().transpose(-1,-2).contiguous()
        desc1 = data["descriptors1"].detach().transpose(-1,-2).contiguous()

        assert desc0.shape[-1] == self.conf.input_dim
        assert desc1.shape[-1] == self.conf.input_dim
        if torch.is_autocast_enabled():
            desc0 = desc0.half()
            desc1 = desc1.half()
        desc0 = self.input_proj(desc0)
        desc1 = self.input_proj(desc1)
        # cache positional embeddings
        encoding0 = self.posenc(kpts0)
        encoding1 = self.posenc(kpts1)
        if torch.isnan(encoding0).any() or torch.isnan(encoding1).any():
            encoding0 = encoding0
            encoding1 = encoding1
            assert 1==2

        # GNN + final_proj + assignment
        all_desc0, all_desc1 = [], []

        for i in range(self.conf.n_layers):
            if self.conf.checkpointed and self.training:
                desc0, desc1 = checkpoint(
                    self.transformers[i], desc0, desc1, encoding0, encoding1, time_embd, adj_mat_fore[...,:-1,:-1]
                )
            else:
                desc0, desc1 = self.transformers[i](desc0, desc1, encoding0, encoding1, time_embd, adj_mat_fore[...,:-1,:-1])
            if self.training or i == self.conf.n_layers - 1:
                all_desc0.append(desc0)
                all_desc1.append(desc1)
                continue  # no early stopping or adaptive width at last layer

        desc0, desc1 = desc0[..., :m, :], desc1[..., :n, :]
        scores, _ = self.log_assignment[i](desc0, desc1)
        m0, m1, mscores0, mscores1 = filter_matches(scores, self.conf.filter_threshold)
        matches, mscores = [], []
        for k in range(b):
            valid = m0[k] > -1
            m_indices_0 = torch.where(valid)[0]
            m_indices_1 = m0[k][valid]
            matches.append(torch.stack([m_indices_0, m_indices_1], -1))
            mscores.append(mscores0[k][valid])


        adj_mat = scores.unsqueeze(1).clone()
        adj_mat[...,:-1,:-1] = (adj_mat[...,:-1,:-1].exp()-0.5)*self.conf.scale
        adj_mat[...,:-1,-1] = (adj_mat[...,:-1,-1].exp()-0.5)*self.conf.scale
        adj_mat[...,-1,:-1] = (adj_mat[...,-1,:-1].exp()-0.5)*self.conf.scale

        return {
            "matches0": m0,
            "matches1": m1,
            "matching_scores0": mscores0,
            "matching_scores1": mscores1,
            "matches": matches,
            "scores": mscores,
            "log_assignment": scores,
            "adj_mat": adj_mat,
        }

    def loss(self, pred, data):
        def loss_params(pred, i):
            la, _ = self.log_assignment[i](
                pred["ref_descriptors0"][:, i], pred["ref_descriptors1"][:, i]
            )
            return {
                "log_assignment": la,
            }

        sum_weights = 1.0
        nll, gt_weights, loss_metrics = self.loss_fn(loss_params(pred, -1), data)
        N = pred["ref_descriptors0"].shape[1]
        losses = {"matcher_total": nll, "last": nll.clone().detach(), **loss_metrics}

        if self.training:
            losses["confidence"] = 0.0

        losses["row_norm"] = pred["log_assignment"].exp()[:, :-1].sum(2).mean(1)

        if self.training:
            for i in range(N):
                params_i = loss_params(pred, i)
                nll, _, _ = self.loss_fn(params_i, data, weights=gt_weights)

                if self.conf.loss.gamma > 0.0:
                    weight = self.conf.loss.gamma ** (N - i)
                else:
                    weight = i + 1
                sum_weights += weight
                losses["matcher_total"] = losses["matcher_total"] + nll * weight

                losses["confidence"] += self.token_confidence[i].loss(
                    pred["ref_descriptors0"][:, i],
                    pred["ref_descriptors1"][:, i],
                    params_i["log_assignment"],
                    pred["log_assignment"],
                ) / (N)

                del params_i
        losses["matcher_total"] /= sum_weights
        # confidences
        if self.training:
            losses["matcher_total"] = losses["matcher_total"] + losses["confidence"]

        if not self.training:
            # add metrics
            metrics = matcher_metrics(pred, data)
        else:
            metrics = {}
        return losses, metrics


__main_model__ = DiffGlue
