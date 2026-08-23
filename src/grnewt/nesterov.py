"""
Anisotropic Nesterov cubic regularization: solve

    eta = (H + (lambda_int / 2) * ||D eta|| * D^2)^{-1} g,             (10)

returning the solution with the largest ||D eta|| (Method 1 of the paper).

The equation is solved through the *secular equation* in the invariant
variables (K, h_hat) of Appendix D, instead of bracketing a function whose
evaluation requires a linear solve.  Consequences:

  * the bracket endpoints are built from closed-form inequalities (P5, P6'
    below), so `scipy.optimize.brentq` is never handed a same-sign bracket;
  * no unbounded `while` loop is used anywhere;
  * no matrix is inverted inside the root-finder;
  * every decision is a function of (K, h_hat), which are exactly invariant
    under the layer-wise affine reparameterization of Appendix E.  The old
    absolute thresholds on lambda_min(H) and on d_i were not.

Conventions
-----------
    c      := lambda_int / 2 > 0
    d      := diag(D), d_i = |order3_i|^(1/3) >= 0
    Z      := {i : d_i = 0} (kernel of D),  R := complement, m := |R|
    M(x)   := H + c x D^2
    phi(x) := ||D M(x)^{-1} g||,   h(x) := phi(x) - x
    r_*    := the root of h;  Method 1's ||D eta_*||.
"""

import time

import scipy.optimize
import torch

__all__ = ["nesterov_lrs", "compute_x0"]


def _as_f64(t):
    return t.to(device="cpu", dtype=torch.float64)


class _Secular:
    """The scalar problem, written in the *shift* variable t := kappa_1 + c x.

    With mu_j := kappa_j - kappa_1 >= 0 (mu_1 = 0) and b = Q^T D_R^{-1} g_hat_R:

        phi(t) = || b / (mu + t) ||,     x(t) = (t - kappa_1) / c,
        h(t)   = phi(t) - x(t),          domain t > t0 := max(0, kappa_1).

    Using t rather than x is not cosmetic.  The root often sits at a relative
    distance ~1e-10 from the pole x0 = max(0, -kappa_1/c); evaluating
    kappa_1 + c x there cancels ~10 significant digits, so phi computed from x
    carries only ~6 correct digits.  In t the smallest denominator *is* the
    variable and is exact.

    Facts used (proofs in the session note):

    P3  ||D eta||^2 = sum_j b_j^2/(mu_j + t)^2 ; M > 0 <=> t > 0.
    P4  phi is non-increasing in t and strictly decreasing where positive, and
        x(t) is strictly increasing, so h is strictly decreasing with
        h -> -inf.  At most one root; exactly one iff h(t0+) > 0.
    P5  if h(t_a) > 0 then t_b := kappa_1 + c phi(t_a) gives h(t_b) <= 0.
    P6' if kappa_1 > 0, t_a := kappa_1 (i.e. x = 0) has h(t_a) = ||b/kappa||;
        if kappa_1 <= 0 and b_1 != 0, t_a := 0.5*min(1, c|b_1|/(1+|kappa_1|))
        has h(t_a) > 0.
    """

    def __init__(self, kappa, b, c):
        # Shift by min(kappa_1, 0), NOT by kappa_1.  The shift exists only to
        # resolve the pole at x0 = -kappa_1/c; when kappa_1 > 0 there is no
        # pole on x >= 0, and shifting by kappa_1 would instead make
        # x(t) = (t - kappa_1)/c cancel catastrophically for small roots
        # (kappa_1 ~ 1e8, c x ~ 1e-3).  Both cancellations are the same
        # phenomenon mirrored; each parameterization avoids exactly one.
        self.k1 = min(float(kappa[0]), 0.0)
        self.mu = kappa - self.k1
        self.b = b
        self.c = float(c)
        self.t0 = 0.0
        self.x0 = max(0.0, -float(kappa[0]) / self.c)
        self.kappa1 = float(kappa[0])

    def x_of_t(self, t):
        return (t - self.k1) / self.c

    def phi(self, t):
        return float(torch.linalg.vector_norm(self.b / (self.mu + t)))

    def h(self, t):
        return self.phi(t) - self.x_of_t(t)

    def dh(self, t):
        den = self.mu + t
        p = float(torch.linalg.vector_norm(self.b / den))
        if p == 0.0:
            return -1.0 / self.c
        return -float((self.b.pow(2) / den.pow(3)).sum()) / p - 1.0 / self.c

    def lower_bracket_point(self):
        """t_a >= t0 with h(t_a) > 0 (P6'); None when the construction fails."""
        b1 = abs(float(self.b[0]))
        if self.kappa1 > 0.0:
            return 0.0                      # x = 0; h(0) = ||b/kappa|| > 0
        if b1 == 0.0:
            return None
        t = 0.5 * min(1.0, self.c * b1 / (1.0 + abs(self.k1)))
        return t if t > 0.0 else None

    def upper_bracket_point(self, t_a):
        """t_b with h(t_b) <= 0 (P5)."""
        return self.k1 + self.c * self.phi(t_a)


def _refine_in_original_basis(H, g, d, c, r, n=4, tol=1e-14):
    """Safeguarded Newton on h_c(x) = ||D M(x)^{-1} g|| - x, evaluated by a
    Cholesky factorization of M(x) = H + c x D^2.

    Why this is needed.  Forming K = D_R^{-1} S D_R^{-1} multiplies the spread
    of the spectrum by cond(D_R)^2, and `eigh` guarantees only an ABSOLUTE
    accuracy eps*||K|| on the eigenvalues; the smallest kappa_j -- hence r and
    eta -- can therefore lose up to log10(cond(D_R)^2) digits even when M(r)
    itself is perfectly conditioned.  Two O(S^3) factorizations recover them.

    Only called when M(r) is formable (no catastrophic cancellation between H
    and c r D^2); returns (r, eta) with ||D eta|| = r to ~eps, or (r, None) if
    no step was accepted.
    """
    D2 = d.pow(2)
    best = None
    for _ in range(n):
        M = H + c * r * torch.diag(D2)
        L, info = torch.linalg.cholesky_ex(M)
        if int(info) != 0:
            break
        v = torch.cholesky_solve(g.unsqueeze(1), L).squeeze(1)
        p = float(torch.linalg.vector_norm(d * v))
        hc = p - r
        if best is None or abs(hc) < abs(best[2]):
            best = (r, v, hc)
        elif abs(hc) >= abs(best[2]):
            break
        if abs(hc) <= tol * max(r, 1.0):
            break
        # phi'(r) = -c (D^2 v)^T M^{-1} (D^2 v) / phi
        if p == 0.0:
            break
        w = D2 * v
        u = torch.cholesky_solve(w.unsqueeze(1), L).squeeze(1)
        dphi = -c * float(w @ u) / p
        denom = dphi - 1.0
        if denom == 0.0:
            break
        r_next = r - hc / denom
        if not (r_next > 0.0) and r_next != 0.0:
            break
        r = max(r_next, 0.0)
    if best is None:
        return r, None
    return best[0], best[1]


def _polish(sec, t, lo, hi, n=3):
    """Safeguarded Newton steps on h; never leaves [lo, hi], never worsens |h|."""
    for _ in range(n):
        ht = sec.h(t)
        dt = sec.dh(t)
        if dt == 0.0:
            break
        cand = t - ht / dt
        if not (lo <= cand <= hi) or abs(sec.h(cand)) >= abs(ht):
            break
        t = cand
    return t


def nesterov_lrs(
    H,
    g,
    order3_,
    *,
    damping_int=1.0,
    force_x0_computation=None,
    threshold_D_sing=0.0,
    hard_case_rtol=1e-12,
    refine=False,
):
    """Learning rates with anisotropic Nesterov cubic regularization.

    Arguments
     * H: summary of the Hessian, symmetric, (S, S)
     * g: summary of the gradient, (S,)
     * order3_: the diagonal d of D, d_i = |order3_i|^(1/3) >= 0
     * damping_int: lambda_int >= 0.  lambda_int == 0 gives the unregularized
       step H^{-1} g and requires H > 0.
     * threshold_D_sing: *relative* threshold under which d_i counts as zero,
       d_i <= threshold_D_sing * max_j d_j.  Default 0.0 = exact zeros only,
       the only choice preserving the affine invariance of Appendix E.
       (In the previous implementation this was an absolute threshold of 1e-5
       compared against |order3_i|^(1/3), i.e. it fired at |order3_i| <= 1e-15,
       and it was not invariant.)
     * hard_case_rtol: relative tolerance defining the near-null eigenspace of
       K + c x0 I in the hard case.
     * refine: run a safeguarded Newton refinement of (r, eta) in the original
       basis when M(r) can be formed without catastrophic cancellation.  OFF by
       default: it is an accuracy optimisation, not part of the correctness of
       the solver, and it is NOT yet validated over the whole fuzz set (see
       STATE.md, open item N3).
     * force_x0_computation: removed; there is a single code path now.

    Returns (lrs, dct_logs).  lrs is None iff dct_logs["found"] is False,
    which happens only when the cubic model is unbounded below (Prop. 2).
    """
    t_begin = time.time()
    if force_x0_computation is not None:
        raise ValueError(
            "force_x0_computation was removed: compute_x0 no longer branches on "
            "'H positive definite' / 'D singular'.  Drop the argument."
        )
    if damping_int < 0:
        raise ValueError(f"damping_int must be >= 0, got {damping_int}.")

    device, dtype = H.device, H.dtype
    logs = {"found": False, "refined": False, "hard_case": False,
            "M_formable": False, "n_ker_D": 0}

    H64, g64, d = _as_f64(H), _as_f64(g), _as_f64(order3_)
    H64 = 0.5 * (H64 + H64.T)
    S = H64.shape[0]

    if bool((d < 0).any()):
        raise ValueError(
            "order3_ must be the diagonal of D, i.e. |order3|^(1/3) >= 0, but it "
            "has negative entries: a caller is passing the raw order-3 summary."
        )

    def _ret(lrs, computation, **extra):
        logs["x0.computation"] = computation
        logs.update(extra)
        logs["time"] = time.time() - t_begin
        if lrs is None:
            return None, logs
        return lrs.to(device=device, dtype=dtype), logs

    # -- trivial: g = 0 ---------------------------------------------------
    if float(torch.linalg.vector_norm(g64)) == 0.0:
        logs["found"] = True
        logs["r"] = torch.zeros((), dtype=torch.float64)
        logs["r_converged"] = True
        logs["x0"] = 0.0
        return _ret(torch.zeros(S, dtype=torch.float64), "trivial_g0")

    c = 0.5 * damping_int

    # -- split on ker(D) --------------------------------------------------
    dmax = float(d.max())
    if dmax > 0:
        zero_mask = (d <= threshold_D_sing * dmax) | (d == 0)
    else:
        zero_mask = torch.ones_like(d, dtype=torch.bool)
    Zi = torch.nonzero(zero_mask, as_tuple=True)[0]
    Ri = torch.nonzero(~zero_mask, as_tuple=True)[0]
    logs["n_ker_D"] = int(Zi.numel())

    # -- degenerate: no cubic term at all ---------------------------------
    if Ri.numel() == 0 or damping_int == 0.0:
        L, info = torch.linalg.cholesky_ex(H64)
        if int(info) != 0:
            return _ret(None, "infeasible_no_cubic", r_converged=False)
        lrs = torch.cholesky_solve(g64.unsqueeze(1), L).squeeze(1)
        logs["found"] = True
        logs["r"] = torch.linalg.vector_norm(d * lrs)
        logs["r_converged"] = True
        logs["x0"] = 0.0
        return _ret(lrs, "no_cubic")

    # -- exact elimination of the ker(D) block (Prop. 1) -------------------
    H_RR = H64[Ri][:, Ri]
    g_R = g64[Ri]
    d_R = d[Ri]
    L_ZZ = H_ZR = None

    if Zi.numel() > 0:
        H_ZZ = H64[Zi][:, Zi]
        H_ZR = H64[Zi][:, Ri]
        L_ZZ, info = torch.linalg.cholesky_ex(H_ZZ)
        if int(info) != 0:
            # Prop. 2: H is not PD on ker(D) => inf T = -inf.  No step exists.
            return _ret(None, "infeasible_kernel", r_converged=False)
        rhs = torch.cat([g64[Zi].unsqueeze(1), H_ZR], dim=1)
        W = torch.cholesky_solve(rhs, L_ZZ)
        Sc = H_RR - H_ZR.T @ W[:, 1:]
        g_hat_R = g_R - H_ZR.T @ W[:, 0]
    else:
        Sc = H_RR
        g_hat_R = g_R

    # -- invariant variables of Appendix D --------------------------------
    K = Sc / (d_R.unsqueeze(1) * d_R.unsqueeze(0))
    K = 0.5 * (K + K.T)
    kappa, Q = torch.linalg.eigh(K)
    b = Q.T @ (g_hat_R / d_R)

    sec = _Secular(kappa, b, c)
    logs["x0"] = sec.x0
    logs["kappa_min"] = float(kappa[0])
    logs["H_pd"] = bool(kappa[0] > 0)     # inertia-based, threshold-free
    logs["hard_case"] = False

    # -- bracket with a proved sign change --------------------------------
    t_a = sec.lower_bracket_point()
    hard = (t_a is None) or (sec.h(t_a) <= 0.0)

    if not hard:
        t_b = sec.upper_bracket_point(t_a)
        h_a, h_b = sec.h(t_a), sec.h(t_b)
        if not (h_a > 0.0 >= h_b and t_b > t_a):
            # Unreachable if P5/P6' hold.  Fail loudly instead of letting
            # scipy raise from inside brentq.
            return _ret(None, "bracket_check_failed", r_converged=False,
                        bracket=(t_a, t_b), h_bracket=(h_a, h_b))
        t = scipy.optimize.brentq(sec.h, t_a, t_b, xtol=1e-300,
                                  rtol=8.9e-16, maxiter=200)
        t = _polish(sec, t, t_a, t_b)
        r = sec.x_of_t(t)
        y = b / (sec.mu + t)
        computation = "secular" if Zi.numel() == 0 else "secular_schur"
    else:
        # -- hard case: b is (numerically) orthogonal to the kappa_1 eigenspace.
        logs["hard_case"] = True
        scale = max(float(kappa.abs().max()), 1.0)
        J = kappa <= kappa[0] + hard_case_rtol * scale
        keep = ~J
        # zeroing b[J] perturbs ||b|| by a relative amount <= hard_case_rtol
        b_ps, kappa_ps = b[keep], kappa[keep]
        x0 = sec.x0
        if keep.sum() == 0:
            L_ps = 0.0
        else:
            L_ps = float(torch.linalg.vector_norm(b_ps / (kappa_ps + c * x0)))
        if L_ps <= x0:
            # Prop. 7: r_* = x0; complete y with a null-space component.
            r = x0
            nJ = max(int(J.sum()), 1)
            y = torch.zeros_like(b)
            if keep.sum() > 0:
                y[keep] = b_ps / (kappa_ps + c * x0)
            y[J] = ((max(x0 * x0 - L_ps * L_ps, 0.0)) ** 0.5) / (nJ ** 0.5)
            computation = "hard_case_boundary"
        else:
            sec_ps = _Secular(kappa_ps, b_ps, c)
            # bracket in the ORIGINAL shift variable of sec_ps
            t_a2 = kappa_ps[0].item() + c * x0
            t_b2 = kappa_ps[0].item() + c * L_ps
            h_a2, h_b2 = sec_ps.h(t_a2), sec_ps.h(t_b2)
            if not (h_a2 > 0.0 >= h_b2 and t_b2 > t_a2):
                return _ret(None, "hard_case_bracket_failed", r_converged=False,
                            bracket=(t_a2, t_b2), h_bracket=(h_a2, h_b2))
            t2 = scipy.optimize.brentq(sec_ps.h, t_a2, t_b2, xtol=1e-300,
                                       rtol=8.9e-16, maxiter=200)
            t2 = _polish(sec_ps, t2, t_a2, t_b2)
            r = sec_ps.x_of_t(t2)
            y = torch.zeros_like(b)
            y[keep] = b_ps / (kappa_ps + c * r)
            computation = "hard_case_interior"

    # -- back to eta ------------------------------------------------------
    eta_R = (Q @ y) / d_R
    lrs = torch.zeros(S, dtype=torch.float64)
    lrs[Ri] = eta_R
    if Zi.numel() > 0:
        lrs[Zi] = torch.cholesky_solve((g64[Zi] - H_ZR @ eta_R).unsqueeze(1),
                                       L_ZZ).squeeze(1)

    # Cross-check in the original basis.  Forming K = D_R^{-1} S D_R^{-1}
    # multiplies the spread of the spectrum by cond(D_R)^2, and `eigh` only
    # guarantees an ABSOLUTE accuracy eps*||K|| on the eigenvalues, so the
    # smallest kappa_j -- hence eta -- can lose up to log10(cond(D_R)^2)
    # digits even when M(r) itself is well conditioned.  One Cholesky solve of
    # M(r) costs O(S^3) and recovers them.
    #
    # The cross-check is attempted only when M(r) can be FORMED at all.  When
    # r sits within a relative ~1e-13 of the pole x0, H and c r D^2 cancel:
    # the entries of M(r) are built from quantities of size G but the result
    # has smallest eigenvalue lam_min << G, so M(r) retains only about
    # log10(lam_min / (eps G)) correct digits.  Requiring lam_min > 1e-10 G
    # keeps ~6 digits.  Below that no evaluation in the original basis means
    # anything, and only the secular solution -- which enforces the fixed
    # point ||D eta|| = r exactly in the reduced variables -- is usable.
    G = float(H64.abs().max()) + c * r * float(d.pow(2).max())
    Mr = H64 + c * r * torch.diag(d.pow(2))
    logs["M_formable"] = float(torch.linalg.eigvalsh(Mr)[0]) > 1e-10 * G
    if refine and logs["M_formable"]:
        r_new, lrs_new = _refine_in_original_basis(H64, g64, d, c, r)
        if lrs_new is not None:
            r, lrs = r_new, lrs_new
            logs["refined"] = True

    logs["found"] = True
    logs["r"] = torch.tensor(r, dtype=torch.float64)
    logs["r_converged"] = True
    logs["lrs"] = lrs
    return _ret(lrs, computation)


def compute_x0(H, order3_, D_squ=None, damping_int=1.0, threshold_D_sing=0.0, **kwargs):
    """x0 = inf{x >= 0 : H + (lambda_int/2) x D^2 > 0}, in closed form.

    Returns (x0, logs).  x0 is None exactly when the set is empty, i.e. when H
    restricted to ker(D) is not positive definite (Prop. 2).  D_squ is accepted
    and ignored (kept for call compatibility).
    """
    logs = {}
    H64, d = _as_f64(H), _as_f64(order3_)
    H64 = 0.5 * (H64 + H64.T)
    c = 0.5 * damping_int
    if c <= 0:
        raise ValueError("damping_int must be > 0 to define x0.")

    dmax = float(d.max())
    zero_mask = ((d <= threshold_D_sing * dmax) | (d == 0)) if dmax > 0 \
        else torch.ones_like(d, dtype=torch.bool)
    Zi = torch.nonzero(zero_mask, as_tuple=True)[0]
    Ri = torch.nonzero(~zero_mask, as_tuple=True)[0]

    if Ri.numel() == 0:
        _, info = torch.linalg.cholesky_ex(H64)
        ok = int(info) == 0
        logs["found"] = ok
        logs["computation"] = "no_cubic"
        return (0.0 if ok else None), logs

    Sc = H64[Ri][:, Ri]
    if Zi.numel() > 0:
        L_ZZ, info = torch.linalg.cholesky_ex(H64[Zi][:, Zi])
        if int(info) != 0:
            logs["found"] = False
            logs["computation"] = "infeasible_kernel"
            return None, logs
        H_ZR = H64[Zi][:, Ri]
        Sc = Sc - H_ZR.T @ torch.cholesky_solve(H_ZR, L_ZZ)

    d_R = d[Ri]
    K = Sc / (d_R.unsqueeze(1) * d_R.unsqueeze(0))
    kappa_min = float(torch.linalg.eigvalsh(0.5 * (K + K.T))[0])
    logs["found"] = True
    logs["computation"] = "secular" if Zi.numel() == 0 else "secular_schur"
    logs["H_pd"] = bool(kappa_min > 0)
    return max(0.0, -kappa_min / c), logs
