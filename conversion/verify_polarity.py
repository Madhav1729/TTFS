"""Event-level check that the polarity TTFS neuron decodes every sign-magnitude level exactly."""

import torch


# ---------------------------------------------------------------- quantisers
def q_sign_magnitude(x, alpha, T):
    """Source quantiser for POLARITY: T magnitudes each side, plus zero."""
    m = torch.round(x.abs() / alpha).clamp(0, T)
    return torch.sign(x) * m * alpha


def q_offset_binary(x, alpha, T):
    """Source quantiser for REFERENCE-TIME: T levels on one ramp through zero."""
    q = torch.round(x / alpha).clamp(-(T // 2), T // 2 - 1)
    return q * alpha


# ------------------------------------------------- event-level neuron models
def polarity_ttfs(V, alpha, T):
    """Polarity neuron with midpoint thresholds alpha*(T+0.5-v); returns (spikes[T], signs[T])."""
    spikes = torch.zeros((T,) + V.shape, dtype=torch.uint8)
    signs = torch.zeros((T,) + V.shape, dtype=torch.int8)
    fired = torch.zeros_like(V, dtype=torch.bool)
    for i in range(T):  # v = i+1
        v = i + 1
        th = alpha * (T + 0.5 - v)
        now = (~fired) & (V.abs() >= th)
        spikes[i] = now.to(torch.uint8)
        signs[i] = torch.where(now, torch.sign(V), torch.zeros_like(V)).to(torch.int8)
        fired |= now
    return spikes, signs


def decode_polarity(spikes, signs, alpha, T):
    out = torch.zeros(spikes.shape[1:], dtype=torch.float32)
    for i in range(T):
        v = i + 1
        mag = (T + 1 - v) * alpha
        hit = spikes[i].bool()
        out = torch.where(hit, signs[i].float() * mag, out)
    return out  # silent stays 0


def offset_binary_ttfs(V, alpha, T):
    """One monotone ramp, midpoint thresholds th(v) = alpha*(T//2 - v - 0.5)."""
    spikes = torch.zeros((T,) + V.shape, dtype=torch.uint8)
    fired = torch.zeros_like(V, dtype=torch.bool)
    for i in range(T):
        v = i + 1
        th = alpha * (T // 2 - v - 0.5)  # decreasing, crosses zero
        now = (~fired) & (V >= th)
        spikes[i] = now.to(torch.uint8)
        fired |= now
    return spikes, fired


def decode_offset_binary(spikes, fired, alpha, T):
    out = torch.full(spikes.shape[1:], float(-(T // 2)) * alpha)  # never fired
    for i in range(T):
        v = i + 1
        val = (T // 2 - v) * alpha
        out = torch.where(spikes[i].bool(), torch.full_like(out, val), out)
    return out


# ------------------------------------------------------------------- checks
def check(name, V, alpha, T, q_fn, sim_fn, dec_fn):
    ref = q_fn(V, alpha, T)
    sim = sim_fn(V, alpha, T)
    got = dec_fn(*sim, alpha, T)
    err = (got - ref).abs().max().item()
    nlev = torch.unique(ref).numel()
    spikes = sim[0]
    ev = spikes.sum(0).float().mean().item()
    ok = "PASS" if err < 1e-5 else "FAIL"
    print(f"  {name:<34} T={T:<3} levels={nlev:<3} max|err|={err:.3e} " f"events/neuron={ev:.3f}  [{ok}]")
    assert spikes.sum(0).max().item() <= 1, "more than one spike per neuron!"
    return err < 1e-5


if __name__ == "__main__":
    torch.manual_seed(0)
    alpha = 0.1
    # heavy-tailed, zero-peaked -- like a real LLM activation
    V = torch.randn(200000) * 0.3
    V[torch.rand_like(V) < 0.02] *= 8.0

    print("event-level verification (no closed form anywhere)\n")
    allok = True
    for T in (4, 8, 16):
        allok &= check(
            "polarity (sign-magnitude)", V, alpha, T, q_sign_magnitude, polarity_ttfs, decode_polarity
        )
    print()
    for T in (8, 16, 32):
        allok &= check(
            "reference-time (offset-binary)",
            V,
            alpha,
            T,
            q_offset_binary,
            offset_binary_ttfs,
            decode_offset_binary,
        )

    print("\nthe claim -- same representable set, half the window:")
    pol = torch.unique(q_sign_magnitude(V, alpha, 8))
    ob = torch.unique(q_offset_binary(V, alpha, 16))
    print(f"  polarity      T= 8 -> {pol.numel():>3} levels, " f"range [{pol.min():+.2f}, {pol.max():+.2f}]")
    print(f"  offset-binary T=16 -> {ob.numel():>3} levels, " f"range [{ob.min():+.2f}, {ob.max():+.2f}]")
    print(f"  step size identical: {alpha}")
    print("\nALL PASS" if allok else "\nSOMETHING FAILED")
