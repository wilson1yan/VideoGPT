import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from
from torch_blocksparse import MatMul, Softmax

# reshapes tensor start from dim i (inclusive)
# to dim j (exclusive) to the desired shape
# e.g. if x.shape = (b, thw, c) then
# view_range(x, 1, 2, (t, h, w)) returns
# x of shape (b, t, h, w, c)
def view_range(x, i, j, shape):
    shape = tuple(shape)

    n_dims = len(x.shape)
    if i < 0:
        i = n_dims + i

    if j is None:
        j = n_dims
    elif j < 0:
        j = n_dims + j

    assert 0 <= i < j <= n_dims

    x_shape = x.shape
    target_shape = x_shape[:i] + shape + x_shape[j:]
    return x.view(target_shape)


def scaled_dot_product_attention(q, k, v, mask=None, attn_dropout=0., training=True):
    # Performs attention over the second to last dimension dn

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


# Only supports self attention
class SparseAttention(nn.Module):
    ops = dict()
    attn_mask = dict()
    block_layout = dict()

    def __init__(self, shape, n_head, causal, num_local_blocks=4, block=32):
        super().__init__()
        self.causal = causal
        self.shape = shape

        self.sparsity_config = StridedSparsityConfig(shape=shape, n_head=n_head,
                                                     causal=causal, block=block,
                                                     num_local_blocks=num_local_blocks)
        self.get_ops()

        if self.shape not in SparseAttention.block_layout:
            SparseAttention.block_layout[self.shape] = self.sparsity_config.make_layout()
        if causal and self.shape not in SparseAttention.attn_mask:
            SparseAttention.attn_mask[self.shape] = self.sparsity_config.make_sparse_attn_mask()

    def get_ops(self):
        if self.shape not in SparseAttention.ops:
            sparsity_layout = self.sparsity_config.make_layout()
            sparse_dot_sdd_nt = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'sdd',
                                       trans_a=False,
                                       trans_b=True)

            sparse_dot_dsd_nn = MatMul(sparsity_layout,
                                       self.sparsity_config.block,
                                       'dsd',
                                       trans_a=False,
                                       trans_b=False)

            sparse_softmax = Softmax(sparsity_layout, self.sparsity_config.block)

            SparseAttention.ops[self.shape] = (sparse_dot_sdd_nt,
                                               sparse_dot_dsd_nn,
                                               sparse_softmax)
        return SparseAttention.ops[self.shape]

    def forward(self, q, k, v, decode_step, decode_idx):
        SparseAttention.block_layout[self.seq_len] = SparseAttention.block_layout[self.seq_len].to(q)
        if self.causal:
            SparseAttention.attn_mask[self.shape] = SparseAttention.attn_mask[self.shape].to(q).type_as(q)
        attn_mask = SparseAttention.attn_mask[self.shape] if self.causal else None

        if q.shape != k.shape or k.shape != v.shape:
            raise Exception('SparseAttention only support self-attention')

        old_shape = q.shape[2:-1]
        q = q.flatten(start_dim=2, end_dim=-2)
        k = k.flatten(start_dim=2, end_dim=-2)
        v = v.flatten(start_dim=2, end_dim=-2)

        if decode_step is not None:
            mask = self.sparsity_config.get_non_block_layout_row(SparseAttention.block_layout[self.seq_len], decode_step)
            out = scaled_dot_product_attention(q, k, v, mask=mask, training=self.training)
        else:
            sparse_dot_sdd_nt, sparse_dot_dsd_nn, sparse_softmax = self.get_ops()
            scaling = float(q.shape[-1]) ** -0.5

            attn_output_weights = sparse_dot_sdd_nt(q, k)
            if attn_mask is not None:
                attn_output_weights = attn_output_weights.masked_fill(attn_mask == 0,
                                                                      float('-inf'))
            attn_output_weights = sparse_softmax(
                attn_output_weights,
                scale=scaling
            )

            out = sparse_dot_dsd_nn(attn_output_weights, v)

        return view_range(out, 2, 3, old_shape)


class StridedSparsityConfig(object):
    """
    Strided Sparse configuration specified in https://arxiv.org/abs/1904.10509 that
    generalizes to arbitrary dimensions
    """
    def __init__(self, shape, n_head, causal, block, num_local_blocks):
        self.n_head = n_head
        self.shape = shape
        self.causal = causal
        self.block = block
        self.num_local_blocks = num_local_blocks

        assert self.num_local_blocks >= 1, 'Must have at least 1 local block'
        assert self.seq_len % self.block == 0, 'seq len must be divisible by block size'

        self._block_shape = self._compute_block_shape()
        self._block_shape_cum = self._block_shape_cum_sizes()

    @property
    def seq_len(self):
        return np.prod(self.shape)

    @property
    def num_blocks(self):
        return self.seq_len // self.block

    def set_local_layout(self, layout):
        num_blocks = self.num_blocks
        for row in range(0, num_blocks):
            end = min(row + self.num_local_blocks, num_blocks)
            for col in range(
                    max(0, row - self.num_local_blocks),
                    (row + 1 if self.causal else end)):
                layout[:, row, col] = 1
        return layout

    def set_global_layout(self, layout):
        num_blocks = self.num_blocks
        n_dim = len(self._block_shape)
        for row in range(num_blocks):
            assert self._to_flattened_idx(self._to_unflattened_idx(row)) == row
            cur_idx = self._to_unflattened_idx(row)
            # no strided attention over last dim
            for d in range(n_dim - 1):
                end = self._block_shape[d]
                for i in range(0, (cur_idx[d] + 1 if self.causal else end)):
                    new_idx = list(cur_idx)
                    new_idx[d] = i
                    new_idx = tuple(new_idx)

                    col = self._to_flattened_idx(new_idx)
                    layout[:, row, col] = 1

        return layout

    def make_layout(self):
        layout = torch.zeros((self.n_head, self.num_blocks, self.num_blocks), dtype=torch.int64)
        layout = self.set_local_layout(layout)
        layout = self.set_global_layout(layout)
        return layout

    def make_sparse_attn_mask(self):
        block_layout = self.make_layout()
        assert block_layout.shape[1] == block_layout.shape[2] == self.num_blocks

        num_dense_blocks = block_layout.sum().item()
        attn_mask = torch.ones(num_dense_blocks, self.block, self.block)
        counter = 0
        for h in range(self.n_head):
            for i in range(self.num_blocks):
                for j in range(self.num_blocks):
                    elem = block_layout[h, i, j].item()
                    if elem == 1:
                        assert i >= j
                        if i == j: # need to mask within block on diagonals
                            attn_mask[counter] = torch.tril(attn_mask[counter])
                        counter += 1
        assert counter == num_dense_blocks

        return attn_mask.unsqueeze(0)

    def get_non_block_layout_row(self, block_layout, row):
        block_row = row // self.block
        block_row = block_layout[:, [block_row]] # n_head x 1 x n_blocks
        block_row = block_row.repeat_interleave(self.block, dim=-1)
        block_row[:, :, row + 1:] = 0.
        return block_row

    ############# Helper functions ##########################

    def _compute_block_shape(self):
        n_dim = len(self.shape)
        cum_prod = 1
        for i in range(n_dim - 1, -1, -1):
            cum_prod *= self.shape[i]
            if cum_prod > self.block:
                break
        assert cum_prod % self.block == 0
        new_shape = (*self.shape[:i], cum_prod // self.block)

        assert np.prod(new_shape) == np.prod(self.shape) // self.block

        return new_shape

    def _block_shape_cum_sizes(self):
        bs = np.flip(np.array(self._block_shape))
        return tuple(np.flip(np.cumprod(bs)[:-1])) + (1,)

    def _to_flattened_idx(self, idx):
        assert len(idx) == len(self._block_shape), f"{len(idx)} != {len(self._block_shape)}"
        flat_idx = 0
        for i in range(len(self._block_shape)):
            flat_idx += idx[i] * self._block_shape_cum[i]
        return flat_idx

    def _to_unflattened_idx(self, flat_idx):
        assert flat_idx < np.prod(self._block_shape)
        idx = []
        for i in range(len(self._block_shape)):
            idx.append(flat_idx // self._block_shape_cum[i])
            flat_idx %= self._block_shape_cum[i]
        return tuple(idx)
