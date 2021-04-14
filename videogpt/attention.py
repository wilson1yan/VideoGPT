import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .utils import shift_dim, view_range, tensor_slice


class AttentionStack(nn.Module):
    def __init__(
        self, shape, embd_dim, n_head, n_layer, dropout,
        attn_dropout, class_cond_dim, frame_cond_shape
    ):
        super().__init__()
        self.shape = shape
        self.embd_dim = embd_dim
        self.use_frame_cond = frame_cond_shape is not None

        self.pos_embd = AddBroadcastPosEmbed(
            shape=shape, embd_dim=embd_dim
        )

        self.attn_nets = nn.ModuleList(
            [
                AttentionBlock(
                    shape=shape,
                    embd_dim=embd_dim,
                    n_head=n_head,
                    n_layer=n_layer,
                    dropout=dropout,
                    attn_dropout=attn_dropout,
                    class_cond_dim=class_cond_dim,
                    frame_cond_shape=frame_cond_shape
                )
                for i in range(n_layer)
            ]
        )

    def forward(self, x, cond, decode_step, decode_idx):
        """
        Args
        ------
            x: (b, d1, d2, ..., dn, embd_dim), where dn-2 = height, dn-1 = width, dn = n_codebooks. n = 4 => (b, t, h, w, l, d) => dim_h = 2, dim_w = 3
            cond: a dictionary of conditioning tensors

            (below is used only when sampling for fast decoding)
            decode: the enumerated rasterscan order of the current idx being sampled
            decode_step: a tuple representing the current idx being sampled
        """
        x = right_shift(x, decode_step)
        x = self.pos_embd(x, decode_step, decode_idx)
        for net in self.attn_nets:
            x = net(x, cond, decode_step, decode_idx)

        return x


class AttentionBlock(nn.Module):
    def __init__(self, shape, embd_dim, n_head, n_layer, dropout,
                 attn_dropout, class_cond_dim, frame_cond_shape):
        super().__init__()
        self.use_frame_cond = frame_cond_shape is not None

        self.pre_attn_norm = LayerNorm(embd_dim, class_cond_dim)
        self.post_attn_dp = nn.Dropout(dropout)
        self.attn = MultiHeadAttention(shape, embd_dim, embd_dim, n_head,
                                       n_layer, causal=True, attn_type='full',
                                       attn_kwargs=dict(attn_dropout=attn_dropout))

        if frame_cond_shape is not None:
            enc_len = np.prod(frame_cond_shape[:-1])
            self.pre_enc_norm = LayerNorm(embd_dim, class_cond_dim)
            self.post_enc_dp = nn.Dropout(dropout)
            self.enc_attn = MultiHeadAttention(shape, embd_dim, frame_cond_shape[-1],
                                               n_head, n_layer, attn_type='full',
                                               attn_kwargs=dict(attn_dropout=0.), causal=False)

        self.pre_fc_norm = LayerNorm(embd_dim, class_cond_dim)
        self.post_fc_dp = nn.Dropout(dropout)
        self.fc_block = nn.Sequential(
            nn.Linear(in_features=embd_dim, out_features=embd_dim * 4),
            GeLU2(),
            nn.Linear(in_features=embd_dim * 4, out_features=embd_dim),
        )

    def forward(self, x, cond, decode_step, decode_idx):
        h = self.pre_attn_norm(x, cond)
        if self.training:
            h = checkpoint(self.attn, h, h, h, decode_step, decode_idx)
        else:
            h = self.attn(h, h, h, decode_step, decode_idx)
        h = self.post_attn_dp(h)
        x = x + h

        if self.use_frame_cond:
            h = self.pre_enc_norm(x, cond)
            h = self.enc_attn(h, cond['frame_cond'], cond['frame_cond'],
                              decode_step, decode_idx)
            h = self.post_enc_dp(h)
            x = x + h

        h = self.pre_fc_norm(x, cond)
        h = self.fc_block(h)
        h = self.post_fc_dp(h)
        x = x + h

        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, shape, dim_q, dim_kv, n_head, n_layer,
                 causal, attn_type, attn_kwargs):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.d_k = dim_q // n_head
        self.d_v = dim_kv // n_head
        self.n_head = n_head

        self.w_qs = nn.Linear(dim_q, n_head * self.d_k, bias=False) # q
        self.w_qs.weight.data.normal_(std=1.0 / np.sqrt(dim_q))

        self.w_ks = nn.Linear(dim_kv, n_head * self.d_k, bias=False) # k
        self.w_ks.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.w_vs = nn.Linear(dim_kv, n_head * self.d_v, bias=False) # v
        self.w_vs.weight.data.normal_(std=1.0 / np.sqrt(dim_kv))

        self.fc = nn.Linear(n_head * self.d_v, dim_q, bias=True) # c
        self.fc.weight.data.normal_(std=1.0 / np.sqrt(dim_q * n_layer))

        if attn_type == 'full':
            self.attn = FullAttention(shape, causal, **attn_kwargs)
        elif attn_type == 'axial':
            assert not causal, 'causal axial attention is not supported'
            self.attn = AxialAttention(len(shape), **attn_kwargs)

        self.cache = None

    def forward(self, q, k, v, decode_step=None, decode_idx=None):
        """ Compute multi-head attention
        Args
            q, k, v: a [b, d1, ..., dn, c] tensor or
                     a [b, 1, ..., 1, c] tensor if decode_step is not None

        Returns
            The output after performing attention
        """

        # compute k, q, v
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        q = view_range(self.w_qs(q), -1, None, (n_head, d_k))
        k = view_range(self.w_ks(k), -1, None, (n_head, d_k))
        v = view_range(self.w_vs(v), -1, None, (n_head, d_v))

        # b x n_head x seq_len x d
        # (b, *d_shape, n_head, d) -> (b, n_head, *d_shape, d)
        q = shift_dim(q, -2, 1)
        k = shift_dim(k, -2, 1)
        v = shift_dim(v, -2, 1)

        # fast decoding
        if decode_step is not None:
            if decode_step == 0:
                if self.causal:
                    k_shape = (q.shape[0], n_head, *self.shape, self.d_k)
                    v_shape = (q.shape[0], n_head, *self.shape, self.d_v)
                    self.cache = dict(k=torch.zeros(k_shape, dtype=k.dtype, device=q.device),
                                    v=torch.zeros(v_shape, dtype=v.dtype, device=q.device))
                else:
                    # cache only once in the non-causal case
                    self.cache = dict(k=k.clone(), v=v.clone())
            if self.causal:
                idx = (slice(None, None), slice(None, None), *[slice(i, i+ 1) for i in decode_idx])
                self.cache['k'][idx] = k
                self.cache['v'][idx] = v
            k, v = self.cache['k'], self.cache['v']

        a = self.attn(q, k, v, decode_step, decode_idx)

        # (b, *d_shape, n_head, d) -> (b, *d_shape, n_head * d)
        a = shift_dim(a, 1, -2).flatten(start_dim=-2)
        a = self.fc(a) # (b x seq_len x embd_dim)

        return a

############## Attention #######################
class FullAttention(nn.Module):
    def __init__(self, shape, causal, attn_dropout):
        super().__init__()
        self.causal = causal
        self.attn_dropout = attn_dropout

        seq_len = np.prod(shape)
        if self.causal:
            self.register_buffer('mask', torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, q, k, v, decode_step, decode_idx):
        mask = self.mask if self.causal else None
        if decode_step is not None and mask is not None:
            mask = mask[[decode_step]]

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        out = scaled_dot_product_attention(q, k, v, mask=mask,
                                           attn_dropout=self.attn_dropout,
                                           training=self.training)

        return view_range(out, 2, 3, old_shape)

class AxialAttention(nn.Module):
    def __init__(self, n_dim, axial_dim):
        super().__init__()
        if axial_dim < 0:
            axial_dim = 2 + n_dim + 1 + axial_dim
        else:
            axial_dim += 2 # account for batch, head, dim
        self.axial_dim = axial_dim

    def forward(self, q, k, v, decode_step, decode_idx):
        q = shift_dim(q, self.axial_dim, -2).flatten(end_dim=-3)
        k = shift_dim(k, self.axial_dim, -2).flatten(end_dim=-3)
        v = shift_dim(v, self.axial_dim, -2)
        old_shape = list(v.shape)
        v = v.flatten(end_dim=-3)

        out = scaled_dot_product_attention(q, k, v, training=self.training)
        out = out.view(*old_shape)
        out = shift_dim(out, -2, self.axial_dim)
        return out

################ Spatiotemporal broadcasted positional embeddings ###############
class AddBroadcastPosEmbed(nn.Module):
    def __init__(self, shape, embd_dim, dim=-1):
        super().__init__()
        assert dim in [-1, 1] # only first or last dim supported
        self.shape = shape
        self.n_dim = n_dim = len(shape)
        self.embd_dim = embd_dim
        self.dim = dim

        assert embd_dim % n_dim == 0, f"{embd_dim} % {n_dim} != 0"
        self.emb = nn.ParameterDict({
             f'd_{i}': nn.Parameter(torch.randn(shape[i], embd_dim // n_dim) * 0.01
                                    if dim == -1 else
                                    torch.randn(embd_dim // n_dim, shape[i]) * 0.01)
             for i in range(n_dim)
        })

    def forward(self, x, decode_step=None, decode_idx=None):
        embs = []
        for i in range(self.n_dim):
            e = self.emb[f'd_{i}']
            if self.dim == -1:
                # (1, 1, ..., 1, self.shape[i], 1, ..., -1)
                e = e.view(1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)), -1)
                e = e.expand(1, *self.shape, -1)
            else:
                e = e.view(1, -1, *((1,) * i), self.shape[i], *((1,) * (self.n_dim - i - 1)))
                e = e.expand(1, -1, *self.shape)
            embs.append(e)

        embs = torch.cat(embs, dim=self.dim)
        if decode_step is not None:
            embs = tensor_slice(embs, [0, *decode_idx, 0],
                                [x.shape[0], *(1,) * self.n_dim, x.shape[-1]])

        return x + embs

################# Helper Functions ###################################
def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
    # Performs scaled dot-product attention over the second to last dimension dn

    # (b, n_head, d1, ..., dn, d)
    attn = torch.matmul(q, k.transpose(-1, -2))
    attn = attn / np.sqrt(q.shape[-1])
    if mask is not None:
        attn = attn.masked_fill(mask == 0, float('-inf'))
    attn_float = F.softmax(attn, dim=-1)
    attn = attn_float.type_as(attn) # b x n_head x d1 x ... x dn x d
    attn = F.dropout(attn, p=attn_dropout, training=training)

    a = torch.matmul(attn, v) # b x n_head x d1 x ... x dn x d

    return a


def right_shift(x, decode_step):
    if decode_step is not None and decode_step > 0:
        return x

    x_shape = list(x.shape)
    x = x.flatten(start_dim=1, end_dim=-2) # (b, seq_len, embd_dim)
    x = F.pad(x[:, :-1], (0, 0, 1, 0))
    x = x.view(*x_shape)

    return x


class GeLU2(nn.Module):
    def forward(self, x):
        return (1.702 * x).sigmoid() * x


class LayerNorm(nn.Module):
    def __init__(self, embd_dim, class_cond_dim):
        super().__init__()
        self.conditional = class_cond_dim is not None

        if self.conditional:
            self.w = nn.Linear(class_cond_dim, embd_dim, bias=False)
            nn.init.constant_(self.w.weight.data, 1. / np.sqrt(class_cond_dim))
            self.wb = nn.Linear(class_cond_dim, embd_dim, bias=False)
        else:
            self.g = nn.Parameter(torch.ones(embd_dim, dtype=torch.float32), requires_grad=True)
            self.b = nn.Parameter(torch.zeros(embd_dim, dtype=torch.float32), requires_grad=True)

    def forward(self, x, cond):
        if self.conditional:  # (b, cond_dim)
            g = 1 + self.w(cond['class_cond']).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1]) # (b, ..., embd_dim)
            b = self.wb(cond['class_cond']).view(x.shape[0], *(1,)*(len(x.shape)-2), x.shape[-1])
        else:
            g = self.g  # (embd_dim,)
            b = self.b

        x_float = x.float()

        mu = x_float.mean(dim=-1, keepdims=True)
        s = (x_float - mu).square().mean(dim=-1, keepdims=True)
        x_float = (x_float - mu) * (1e-5 + s.rsqrt())  # (b, ..., embd_dim)
        x_float = x_float * g + b

        x = x_float.type_as(x)
        return x
