"""
Cox partial likelihood loss functions.

Two variants:
  cox_loss_breslow — O(n log n) Breslow approximation via sort-cumsum.
                     Used for E1–E9 (CNN ablation).

  cox_loss_efron   — Efron correction for tied event times.
                     Lung 1 records survival in integer days → many ties.
                     Used for E10–E17 (clinical, spatial, pretrained, COLIPRI).

Mathematical note
-----------------
Cox partial negative log-likelihood for M patients sorted descending:

  L = -1/|ε| × Σ_{i∈ε} [ η(x_i) − log Σ_{j: t_j ≥ t_i} exp(η(x_j)) ]

where ε = {i : Δ_i = 1} are uncensored patients and η(x_i) is the
predicted log-hazard for patient i.

Normalisation by |ε| keeps gradient magnitude consistent across mini-batches
with different censoring fractions.

Sign convention
---------------
All models produce log-hazard ratios.  Higher log-hazard = higher risk =
shorter expected survival.  concordance_index() from lifelines expects higher
predicted score = longer survival, so we negate: pass -log_h.

References
----------
Breslow (1974). Covariance analysis of censored survival data.
Efron (1977). The efficiency of Cox's likelihood function for censored data.
"""

import torch


def cox_loss_breslow(
    log_h: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Cox partial negative log-likelihood, Breslow approximation.

    The sort-then-cumsum trick converts the O(M²) double loop to O(M log M):
    after sorting descending, the risk set for position k is positions k…M,
    so log(Σ_{risk set k}) = log(cumsum(exp(η))[k]).

    Parameters
    ----------
    log_h : (B,) tensor — predicted log-hazard ratios
    t     : (B,) tensor — survival times
    e     : (B,) tensor — event indicators (1=event, 0=censored)
    eps   : float — numerical stability constant for log

    Returns
    -------
    Scalar loss (negative partial log-likelihood / |events|).
    Returns 0 (with gradient) if there are no events in the batch.
    """
    order = torch.argsort(t, descending=True)
    log_h = log_h[order]
    e     = e[order].float()
    # log of cumulative risk set sums
    lcs   = torch.log(torch.cumsum(torch.exp(log_h), dim=0) + eps)
    n_ev  = e.sum()
    if n_ev < 1:
        return torch.tensor(0.0, requires_grad=True, device=log_h.device)
    return -((log_h - lcs) * e).sum() / n_ev


def cox_loss_efron(
    log_h: torch.Tensor,
    t: torch.Tensor,
    e: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """
    Cox partial negative log-likelihood, Efron approximation for ties.

    For each unique event time t* with n_tied simultaneous deaths,
    the Efron denominator for the j-th (j=0..n_tied-1) tied patient is:

      D_j = Σ_{at risk at t*} exp(η)  −  (j/n_tied) × Σ_{tied at t*} exp(η)

    This averages the contribution of tied patients across positions,
    correcting the Breslow overestimate of the risk set for tied events.

    Parameters
    ----------
    log_h : (B,) tensor — predicted log-hazard ratios
    t     : (B,) tensor — survival times
    e     : (B,) tensor — event indicators (1=event, 0=censored)
    eps   : float — numerical stability constant

    Returns
    -------
    Scalar loss normalised by total number of events.
    Returns 0 (with gradient) if there are no events in the batch.
    """
    log_h  = log_h.float()
    t      = t.float()
    e      = e.float()
    utimes = torch.unique(t[e == 1], sorted=True)

    if len(utimes) == 0:
        return torch.tensor(0.0, requires_grad=True, device=log_h.device)

    loss = torch.tensor(0.0, device=log_h.device, dtype=torch.float32)
    n_ev = 0

    for ti in utimes:
        at_risk  = t >= ti
        tied     = (t == ti) & (e == 1)
        n_tied   = int(tied.sum().item())
        if n_tied == 0:
            continue
        risk_exp = torch.exp(log_h[at_risk]).sum()
        tied_exp = torch.exp(log_h[tied]).sum()
        tied_lh  = log_h[tied].sum()
        for j in range(n_tied):
            denom = risk_exp - (j / n_tied) * tied_exp + eps
            loss  = loss - tied_lh / n_tied + torch.log(denom)
        n_ev += n_tied

    return loss / max(n_ev, 1)


LOSS_REGISTRY = {
    "breslow": cox_loss_breslow,
    "efron":   cox_loss_efron,
}
