import math
from typing import Iterable, Optional, Callable

import torch
from torch.optim import Optimizer


class ThermoLion(Optimizer):
    r"""ThermoLion optimizer.

    A thermodynamic-enhanced variant of Lion that interpolates between
    sign-based updates and variance-normalised (Adam-like) updates,
    with an annealed exploration term.

    The implementation is based on the experimental code you used in your
    benchmark script, with some minor clean-ups to follow the standard
    PyTorch :class:`Optimizer` API.

    Args:
        params: Iterable of parameters to optimize.
        lr: Base learning rate.
        betas: Tuple of (beta1, beta2) coefficients for the first and
            second moment estimates.
        temp_decay: Multiplicative decay factor for the temperature
            controlling the stochastic exploration term. Values in
            [0.9, 0.999] are typical.
        weight_decay: Decoupled weight decay factor.

    Notes:
        • The optimizer maintains first- and second-moment estimates
          ``m`` and ``v`` similar to Adam.
        • A signal-to-noise ratio (SNR) gate interpolates between
          a Lion-style sign update and a variance-normalised update.
        • A temperature scalar controls an additive Gaussian exploration
          term that decays over time.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas = (0.9, 0.99),
        temp_decay: float = 0.99,
        weight_decay: float = 0.01,
        eps: float = 1e-8,
    ) -> None:
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if not 0.0 < temp_decay <= 1.0:
            raise ValueError(f"Invalid temp_decay value: {temp_decay}")
        if eps <= 0.0:
            raise ValueError(f"Invalid eps value: {eps}")

        defaults = dict(
            lr=lr,
            betas=betas,
            temp_decay=temp_decay,
            weight_decay=weight_decay,
            eps=eps,
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None):
        """Performs a single optimization step.

        Args:
            closure: A closure that re-evaluates the model and returns
                the loss. This is kept for API compatibility and is
                rarely needed in practice.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            tdec = group["temp_decay"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("ThermoLion does not support sparse gradients")

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state["step"] = 0
                    state["m"] = torch.zeros_like(p)
                    state["v"] = torch.zeros_like(p)
                    state["temp"] = 1.0

                m = state["m"]
                v = state["v"]
                temp = state["temp"]

                state["step"] += 1
                temp *= tdec
                state["temp"] = temp

                # First and second moments
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # Signal-to-noise ratio gate
                snr = m.abs() / (v.sqrt() + eps)
                gate = torch.tanh(snr)

                # Mixed update: Lion-style sign term + variance-normalised term
                sign_m = torch.sign(m)
                align = torch.clamp(sign_m * torch.sign(g), 0.0, 1.0)

                lion_step = sign_m * (1.0 + 0.5 * align)
                adam_like_step = m / (v.sqrt() + eps)

                step = (1.0 - gate) * lion_step * lr + gate * adam_like_step * lr * 2.0

                # Annealed stochastic exploration (Gaussian)
                if temp > 0.01:
                    # Scale noise by global variance level and temperature
                    noise_std = math.sqrt(max(temp * float(v.mean()), 0.0) + 1e-10)
                    step = step + torch.randn_like(p) * (noise_std * lr * (1.0 - gate))

                # Decoupled weight decay
                if wd > 0.0:
                    p.mul_(1.0 - lr * wd)

                # Parameter update
                p.add_(step, alpha=-1.0)

        return loss
