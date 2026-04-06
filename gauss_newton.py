"""
Gauss-Newton Optimizer for parameter-golf.

Implements the blended GN-Frank-Wolfe update (senmiao_gn variant):
  1. Compute blended gradient (tilde_g): g_t + lr*α * H_t * v_{t-1}
  2. LMO: feed tilde_g into Muon (Newton-Schulz) for 2D matrices, Adam for the rest
     (each LMO maintains its own internal momentum, outputs a direction without lr)
  3. Convex combination: v_t = β * v_{t-1} + (1-β) * s_t
  4. Apply: param += group_lr * v_t   (per-group learning rate)

The caller updates self.group_lrs with scheduled values each step before
calling update_direction / step.  The second-order term uses
group_lrs['matrix'] as reference lr.

Reference: nanogpt-gn-senmiao/Inner_Solver.py
"""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor
from torch.func import functional_call, jvp


def _zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Newton-Schulz orthogonalization for 2D matrices (same as train_gpt.py)."""
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if transposed else X


class GaussNewtonOptimizer:
    """
    Gauss-Newton optimizer using blended GN-Frank-Wolfe updates.

    Works with the *uncompiled* base_model via functional_call + jvp.
    Does not inherit from torch.optim.Optimizer.
    """

    def __init__(
        self,
        base_model: torch.nn.Module,
        param_info: list[dict[str, Any]],
        *,
        beta: float = 0.9,
        so_ratio: float = 0.1,
        group_lrs: dict[str, float] | None = None,
        muon_momentum: float = 0.95,
        muon_backend_steps: int = 5,
        adam_betas: tuple[float, float] = (0.9, 0.95),
        adam_eps: float = 1e-8,
        vocab_size: int = 1024,
    ):
        """
        Args:
            base_model: Uncompiled GPT model.
            param_info: List of dicts, one per parameter, with keys:
                'name': str, 'param': Tensor, 'group': str in {'matrix','scalar','embed','head'}
            beta: FW convex-combination coefficient.
            so_ratio: α, weight of second-order Hessian term.
            group_lrs: Per-group learning rates, e.g.
                {'matrix': 0.04, 'scalar': 0.04, 'embed': 0.05, 'head': 0.008}.
                The caller should update self.group_lrs with scheduled values
                each step.  The 'matrix' lr is also used as reference for the
                second-order term (lr*α).
            muon_momentum: Momentum coefficient for Muon LMO (matches MUON_MOMENTUM).
                The caller can update self.muon_momentum each step for warmup.
            muon_backend_steps: Newton-Schulz iterations for Muon LMO.
            adam_betas: (β1, β2) for Adam LMO (matches BETA1, BETA2).
            adam_eps: Epsilon for Adam LMO (matches ADAM_EPS).
            vocab_size: Model vocab size (needed for one-hot encoding).
        """
        self.base_model = base_model
        self.beta = beta
        self.so_ratio = so_ratio
        self.group_lrs = group_lrs or {'matrix': 0.04, 'scalar': 0.04, 'embed': 0.05, 'head': 0.008}
        self.muon_momentum = muon_momentum
        self.muon_backend_steps = muon_backend_steps
        self.adam_beta1, self.adam_beta2 = adam_betas
        self.adam_eps = adam_eps
        self.vocab_size = vocab_size

        # Parameter metadata
        self.param_names: list[str] = [info['name'] for info in param_info]
        self.params: list[Tensor] = [info['param'] for info in param_info]
        self.groups: list[str] = [info['group'] for info in param_info]

        # Direction vectors v_t (persistent across steps for FW momentum)
        self.direction: list[Tensor] = [torch.zeros_like(p) for p in self.params]

        # Accumulated blended gradient for current outer step
        self.grad_accum: list[Tensor] = [torch.zeros_like(p) for p in self.params]
        self.accum_count: int = 0

        # Muon momentum buffers (for matrix params only)
        self.muon_buf: dict[int, Tensor] = {}
        for i, g in enumerate(self.groups):
            if g == 'matrix':
                self.muon_buf[i] = torch.zeros_like(self.params[i])

        # Adam state (for non-matrix params)
        self.adam_state: dict[int, dict[str, Tensor | int]] = {}
        for i, g in enumerate(self.groups):
            if g != 'matrix':
                self.adam_state[i] = {
                    'exp_avg': torch.zeros_like(self.params[i]),
                    'exp_avg_sq': torch.zeros_like(self.params[i]),
                    'step': 0,
                }

        # Outer params (detached copies with grad tracking, refreshed each outer step)
        self.outer_params: tuple[Tensor, ...] = ()

    def reset_outer_params(self) -> None:
        """Detach current model params and prepare for a new outer step."""
        self.outer_params = tuple(
            p.detach().requires_grad_(True) for p in self.params
        )
        # Reset gradient accumulator
        for g in self.grad_accum:
            g.zero_()
        self.accum_count = 0

    @torch.no_grad()
    def _average_across_dp(self, tensors: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        """All-reduce gradient tensors across DDP ranks."""
        if not (dist.is_available() and dist.is_initialized()):
            return tensors
        world_size = dist.get_world_size()
        if world_size <= 1:
            return tensors
        averaged = []
        for t in tensors:
            synced = t.clone()
            dist.all_reduce(synced, op=dist.ReduceOp.SUM)
            synced.div_(world_size)
            averaged.append(synced)
        return tuple(averaged)

    def update_direction(self, input_ids: Tensor, target_ids: Tensor) -> None:
        """
        Compute blended GN gradient for one micro-batch and accumulate.

        This computes: g_t + lr*α * J^T * H * J * v_{t-1}
        where:
          - g_t = J^T * (p - y) / (B*S)  [first-order gradient]
          - H = diag(p) - pp^T  [Hessian of CE w.r.t. logits]
          - J = Jacobian of logits w.r.t. params
          - v_{t-1} = self.direction (momentum)
          - lr = self.lr (set by caller each step)
        """
        param_names = self.param_names
        outer_params = self.outer_params

        # functional forward that returns logits
        def model_fn(params_tuple):
            param_dict = {name: p for name, p in zip(param_names, params_tuple)}
            return functional_call(
                self.base_model, param_dict, args=(input_ids,),
                kwargs={'return_logits': True},
            )

        # Enable JVP attention on all attention modules (uses fast JVPAttn kernel
        # instead of the slow MATH backend for forward-mode AD compatibility).
        for module in self.base_model.modules():
            if hasattr(module, 'use_jvp_attn'):
                module.use_jvp_attn = True
        try:
            logits, jvp_product = jvp(model_fn, (outer_params,), (tuple(self.direction),))
        finally:
            for module in self.base_model.modules():
                if hasattr(module, 'use_jvp_attn'):
                    module.use_jvp_attn = False
        jvp_product = jvp_product.detach()

        batch_size, seq_len, vocab_size = logits.shape

        # Softmax probabilities (detached, float32 for precision)
        with torch.no_grad():
            p = F.softmax(logits.float(), dim=-1).detach()

        # Hessian-vector product: H * Jv = diag(p)*Jv - p*(p^T * Jv)
        jvp_f = jvp_product.float()
        p_dot_jvp = p * jvp_f  # (B, S, V)
        pT_dot_jvp = (p * jvp_f).sum(dim=-1, keepdim=True)  # (B, S, 1)
        hvp = p_dot_jvp - p * pT_dot_jvp  # (B, S, V)

        # Blended VJP input: matrix lr * α for second-order term
        eta_alpha = self.group_lrs['matrix'] * self.so_ratio
        labels_one_hot = F.one_hot(target_ids, num_classes=vocab_size).float()
        vjp_input = ((p - labels_one_hot) + eta_alpha * hvp) / (batch_size * seq_len)

        # VJP: J^T @ vjp_input -> parameter-space blended gradient
        blended_grad = torch.autograd.grad(
            outputs=logits,
            inputs=outer_params,
            grad_outputs=vjp_input.to(logits.dtype),
            retain_graph=False,
            create_graph=False,
        )

        # All-reduce across DDP ranks
        blended_grad = self._average_across_dp(blended_grad)

        # Accumulate
        with torch.no_grad():
            for i, g in enumerate(blended_grad):
                self.grad_accum[i].add_(g)
        self.accum_count += 1

    @torch.no_grad()
    def step(self) -> None:
        """
        Apply one outer GN step:
        1. Average accumulated gradients (tilde_g)
        2. Feed tilde_g into LMO (Muon / Adam with internal momentum)
        3. Convex combination v_t = β*v_prev + (1-β)*s_t
        4. Write updated params: param += lr * v_t
        """
        beta = self.beta

        # Average accumulated gradients over micro-batches
        if self.accum_count > 1:
            for g in self.grad_accum:
                g.div_(self.accum_count)

        # Save v_{t-1}
        v_prev = [d.clone() for d in self.direction]

        for i in range(len(self.params)):
            tilde_g = self.grad_accum[i]

            # 1. Compute LMO direction s_t (tilde_g fed directly, no EMA)
            if self.groups[i] == 'matrix':
                s_t = self._muon_lmo(tilde_g, i)
            else:
                s_t = self._adam_lmo(tilde_g, i)

            # 2. Convex combination: v_t = β * v_prev + (1-β) * s_t
            self.direction[i].copy_(beta * v_prev[i] + (1 - beta) * s_t)

            # 3. Apply to model params: param += group_lr * v_t
            group_lr = self.group_lrs[self.groups[i]]
            self.params[i].data.add_(self.direction[i], alpha=group_lr)

    def _muon_lmo(self, tilde_g: Tensor, idx: int) -> Tensor:
        """Muon LMO: Newton-Schulz orthogonalization with Nesterov momentum.
        Returns a direction (no lr scaling); lr is applied at param update."""
        buf = self.muon_buf[idx]
        momentum = self.muon_momentum

        # Momentum update (standard SGD momentum: buf = β*buf + g)
        buf.mul_(momentum).add_(tilde_g)

        # Nesterov: g_eff = g + β * buf
        g_eff = tilde_g.add(buf, alpha=momentum)

        # Newton-Schulz orthogonalization
        normalized = _zeropower_via_newtonschulz5(g_eff, steps=self.muon_backend_steps)

        # Scale correction (same as original Muon)
        lr_ratio = max(1, g_eff.size(0) / g_eff.size(1)) ** 0.5

        return -lr_ratio * normalized.to(dtype=tilde_g.dtype)

    def _adam_lmo(self, tilde_g: Tensor, idx: int) -> Tensor:
        """Adam LMO for scalar/embedding/head parameters.
        Returns a direction (no lr scaling); lr is applied at param update."""
        state = self.adam_state[idx]
        beta1 = self.adam_beta1
        beta2 = self.adam_beta2
        eps = self.adam_eps

        state['step'] += 1
        step = state['step']

        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']

        exp_avg.mul_(beta1).add_(tilde_g, alpha=1 - beta1)
        exp_avg_sq.mul_(beta2).addcmul_(tilde_g, tilde_g, value=1 - beta2)

        m_hat = exp_avg / (1 - beta1 ** step)
        v_hat = exp_avg_sq / (1 - beta2 ** step)

        return -m_hat / (v_hat.sqrt() + eps)
