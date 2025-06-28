import torch
def _orthogonalize_via_ns(G, steps: int=5):
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use aAdd commentMore actions
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.
    """
    assert G.ndim >= 2 # batched Muon implementation by @scottjmaddox, and put into practice in the record by @YouJiacheng
    a, b, c = (3.4445, -4.7750,  2.0315)
    X = G.to(torch.float16)
    if G.size(-2) > G.size(-1):
        X = X.mT

    # Ensure spectral norm is at most 1
    X = X / (X.norm(dim=(-2, -1), keepdim=True) + 1e-7)
    # Perform the NS iterations
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * A @ A # quintic computation strategy adapted from suggestion by @jxbz, @leloykun, and @YouJiacheng
        X = a * X + B @ X
    
    if G.size(-2) > G.size(-1):
        X = X.mT
    return X

def spectral_hardcap(W: torch.Tensor, beta: float = 1.0, ns_steps: int = 5):
    """
    Hard-caps the singular values of a matrix W to be at most beta.
    """
    if flip := (W.shape[0] > W.shape[1]):
        W = W.T

    W_fp16 = W.to(torch.float16)
    OW = _orthogonalize_via_ns(W_fp16, steps=ns_steps)
    aW = beta * OW - W_fp16
    result = 0.5 * (beta * OW + W_fp16 - aW @ _orthogonalize_via_ns(aW, steps=ns_steps).T @ OW)
    result = result.to(W.dtype)

    if flip:
        result = result.T
    return result
