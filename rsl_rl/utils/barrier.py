import torch

def _as_tensor(x, dtype=None, device=None):
    if isinstance(x, torch.Tensor):
        return x
    return torch.tensor(x, dtype=dtype, device=device)

def relaxed_log_barrier_one_sided(z, delta, k=2, eps_log=1e-12):
    """
    Torch version of the one-sided relaxed log barrier (Feller & Ebenbauer β_k).
    - z: tensor or convertible (margin; feasible if z > 0)
    - delta: scalar or tensor (transition point)
    - k: integer >= 2, polynomial degree for β_k
    Returns:
      tensor same shape as z, differentiable.
    """
    # convert inputs to tensors on same device/dtype
    z = _as_tensor(z)
    device = z.device
    dtype = z.dtype
    delta = _as_tensor(delta, dtype=dtype, device=device).to(dtype=dtype)

    assert float(delta) > 0.0, "delta must be > 0"
    assert int(k) >= 2, "k must be >= 2"

    # mask for interior region (use log)
    mask = z > delta

    # safe interior log: clip above tiny positive to avoid log(0)
    # but ensure clipping doesn't break grad too much: use clamp_min
    z_safe = torch.clamp(z, min=eps_log)
    interior = -torch.log(z_safe)

    # exterior: β_k(z; delta) = (k-1)/k * [ ((z - kδ) / ((k-1)δ) )^k - 1 ] - ln(δ)
    # compute elementwise (works with negative base if k integer)
    zk = z
    num = (zk - (k * delta))
    den = ((k - 1) * delta)
    # avoid division by zero in pathological delta ~ 0 (we asserted delta>0)
    ratio = num / den
    ratio_pow = torch.pow(ratio, k)  # integer k ok for negative ratio
    beta_k = ((k - 1) / k) * (ratio_pow - 1.0) - torch.log(delta.clamp_min(eps_log))

    # choose per-element value with torch.where to keep autograd
    B = torch.where(mask, interior, beta_k)
    return B


def relaxed_barrier_for_interval(x, lower=None, upper=None,
                                       delta_frac=0.1, k=2, constraint_range=None):
    """
    Interval barrier using two one-sided relaxed log barriers.
    - x: tensor (can be any shape)
    - lower, upper: floats/tensors or None. If provided, the corresponding one-sided margin is computed.
      margin for lower: s_low = x - lower  (feasible if >=0)
      margin for upper: s_up = upper - x    (feasible if >=0)
    - delta_frac: fraction of constraint_range to set delta = delta_frac * range
    - k: polynomial degree for β_k
    - constraint_range: optional absolute range (float or tensor). If None and both lower/upper exist,
        range = max(upper-lower, 1e-6). If only one bound provided, fallback to 1.0 scale.
    Returns:
      tensor same shape as x: sum of barrier contributions (>= ~ -ln(delta) smallest)
    """
    x = _as_tensor(x)
    device = x.device
    dtype = x.dtype

    # determine scale/range for delta
    if constraint_range is None:
        if (lower is not None) and (upper is not None):
            lower_t = _as_tensor(lower, dtype=dtype, device=device)
            upper_t = _as_tensor(upper, dtype=dtype, device=device)
            rng = (upper_t - lower_t).clamp_min(1e-6)
        else:
            # fallback scale: 1.0 (user can override with constraint_range)
            rng = _as_tensor(1.0, dtype=dtype, device=device)
    else:
        rng = _as_tensor(constraint_range, dtype=dtype, device=device).clamp_min(1e-6)

    delta = (delta_frac * rng).clamp_min(1e-8)  # ensure positive; broadcastable
    B_total = torch.zeros_like(x, dtype=dtype, device=device)

    if lower is not None:
        lower_t = _as_tensor(lower, dtype=dtype, device=device)
        s_low = x - lower_t
        B_low = relaxed_log_barrier_one_sided(s_low, delta=delta, k=k)
        B_total = B_total + B_low

    if upper is not None:
        upper_t = _as_tensor(upper, dtype=dtype, device=device)
        s_up = upper_t - x
        B_up = relaxed_log_barrier_one_sided(s_up, delta=delta, k=k)
        B_total = B_total + B_up

    return B_total