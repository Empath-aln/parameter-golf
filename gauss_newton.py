"""
Gauss-Newton Optimizer for parameter-golf.

Implements the blended GN-Frank-Wolfe update (senmiao_gn variant):
  1. Compute blended gradient: g_t + η*α * H_t * v_{t-1}
  2. Update EMA: m_t = β * m_{t-1} + (1-β) * blended_gradient
  3. LMO: Muon (Newton-Schulz) for 2D matrices, Adam for the rest
  4. Convex combination: v_t = β * v_{t-1} + (1-β) * s_t
  5. Apply: param += lr * v_t

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
        inner_lr: float = 1e-3,
        muon_backend_steps: int = 5,
        vocab_size: int = 1024,
        ref_lr: float = 0.04,
    ):
        """
        Args:
            base_model: Uncompiled GPT model.
            param_info: List of dicts, one per parameter, with keys:
                'name': str, 'param': Tensor, 'group': str in {'matrix','scalar','embed','head'}
            beta: EMA momentum coefficient.
            so_ratio: α, weight of second-order Hessian term.
            inner_lr: Learning rate for the inner LMO (direction solver).
            muon_backend_steps: Newton-Schulz iterations for Muon LMO.
            vocab_size: Model vocab size (needed for one-hot encoding).
            ref_lr: Reference outer LR used for η in η*α computation.
        """
        self.base_model = base_model
        self.beta = beta
        self.so_ratio = so_ratio
        self.inner_lr = inner_lr
        self.muon_backend_steps = muon_backend_steps
        self.vocab_size = vocab_size
        self.ref_lr = ref_lr

        # Parameter metadata
        self.param_names: list[str] = [info['name'] for info in param_info]
        self.params: list[Tensor] = [info['param'] for info in param_info]
        self.groups: list[str] = [info['group'] for info in param_info]

        n = len(self.params)

        # Direction vectors v_t (persistent across steps for momentum)
        self.direction: list[Tensor] = [torch.zeros_like(p) for p in self.params]

        # EMA buffers m_t
        self.ema_m: list[Tensor] = [torch.zeros_like(p) for p in self.params]

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

        This computes: g_t + η*α * J^T * H * J * v_{t-1}
        where:
          - g_t = J^T * (p - y) / (B*S)  [first-order gradient]
          - H = diag(p) - pp^T  [Hessian of CE w.r.t. logits]
          - J = Jacobian of logits w.r.t. params
          - v_{t-1} = self.direction (momentum)
        """
        param_names = self.param_names
        outer_params = self.outer_params

        # functional forward that returns logits
        # We use the forward_logits method via a wrapper module
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

        # Blended VJP input
        eta_alpha = self.ref_lr * self.so_ratio
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
    def step(self, lr_scale: float = 1.0) -> None:
        """
        Apply one outer GN step:
        1. Average accumulated gradients
        2. Update EMA m_t
        3. Compute LMO direction s_t
        4. Convex combination v_t = β*v_prev + (1-β)*s_t
        5. Write updated params
        """
        beta = self.beta

        # Average accumulated gradients over micro-batches
        if self.accum_count > 1:
            for g in self.grad_accum:
                g.div_(self.accum_count)

        # Save v_{t-1}
        v_prev = [d.clone() for d in self.direction]

        for i in range(len(self.params)):
            grad = self.grad_accum[i]

            # 1. EMA update: m_t = β * m_{t-1} + (1-β) * grad
            self.ema_m[i].lerp_(grad, 1 - beta)
            m_t = self.ema_m[i]

            # 2. Compute LMO direction s_t
            if self.groups[i] == 'matrix':
                s_t = self._muon_lmo(m_t, i)
            else:
                s_t = self._adam_lmo(m_t, i)

            # 3. Convex combination: v_t = β * v_prev + (1-β) * s_t
            self.direction[i].copy_(beta * v_prev[i] + (1 - beta) * s_t)

            # 4. Apply to model params: param += lr_scale * v_t
            # The direction IS the update (inner_lr is already baked into s_t)
            self.params[i].data.add_(self.direction[i], alpha=lr_scale)

    def _muon_lmo(self, m_t: Tensor, idx: int) -> Tensor:
        """Muon LMO: Newton-Schulz orthogonalization with Nesterov momentum."""
        buf = self.muon_buf[idx]
        muon_beta = 0.95

        # Momentum update
        buf.lerp_(m_t, 1 - muon_beta)

        # Nesterov: g_eff = m_t + momentum * buf
        g_eff = m_t + muon_beta * buf

        # Newton-Schulz orthogonalization
        normalized = _zeropower_via_newtonschulz5(g_eff, steps=self.muon_backend_steps)

        # Scale correction
        lr_ratio = max(1, g_eff.size(0) / g_eff.size(1)) ** 0.5

        return -self.inner_lr * lr_ratio * normalized.to(dtype=m_t.dtype)

    def _adam_lmo(self, m_t: Tensor, idx: int) -> Tensor:
        """Adam LMO for scalar/embedding/head parameters."""
        state = self.adam_state[idx]
        adam_beta1 = 0.9
        adam_beta2 = 0.95
        adam_eps = 1e-8

        state['step'] += 1
        step = state['step']

        exp_avg = state['exp_avg']
        exp_avg_sq = state['exp_avg_sq']

        exp_avg.mul_(adam_beta1).add_(m_t, alpha=1 - adam_beta1)
        exp_avg_sq.mul_(adam_beta2).addcmul_(m_t, m_t, value=1 - adam_beta2)

        m_hat = exp_avg / (1 - adam_beta1 ** step)
        v_hat = exp_avg_sq / (1 - adam_beta2 ** step)

        return -self.inner_lr * m_hat / (v_hat.sqrt() + adam_eps)
