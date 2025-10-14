# optimization_dual.py
import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

class PureDualPersuitOptimization:
    """
    Pure-dual (robust) layer, DCP-compliant and CVXPY 2D-variable safe.
    - Parameters to the layer: A ∈ R^{d_xi x d_k}, b ∈ R^{d_xi}, D ∈ R^{T x V}
    - Decision variables (all 2D):
        mu ∈ R^{N x (V+1)}, lambda ∈ R^{N x T},
        pi ∈ R^{(N*T) x V}, nu ∈ R^{(N*T) x V},
        Xi ∈ R^{(N*V) x (d_xi*(V+1))} (nonneg)
    - Outputs: mapped_values, loss
    """

    ACTION_FORWARD = 0
    ACTION_TURN_LEFT = 1
    ACTION_TURN_RIGHT = 2
    ACTION_FOUND = 3

    def __init__(self, batch_size, T, edge, N, Ld, dxi=None, gamma=0.95,
                 device="cpu", dtype=torch.float32):
        self.batch_size = batch_size
        self.T = T
        self.edge = edge
        self.V = edge * edge
        self.N = N
        self.Ld = Ld
        self.dk = self.V + self.Ld
        self.dxi = (2 * self.Ld + 1) if dxi is None else dxi
        self.gamma = gamma
        self.device = device
        self.dtype = dtype

        # Constants
        S_np = np.hstack([np.zeros((self.V, 1)), np.eye(self.V)])
        E_np = np.ones((1, self.V + 1))
        H_np = np.hstack([np.eye(self.V), np.zeros((self.V, self.Ld))])
        C_np = H_np.T @ S_np
        self.S = cp.Constant(S_np)
        self.E = cp.Constant(E_np)
        self.H = cp.Constant(H_np)
        self.C = cp.Constant(C_np)

        self.Gamma = [
            cp.Constant(np.hstack([[gamma**t], np.zeros(self.V)]).reshape(1, self.V + 1))
            for t in range(self.T)
        ]
        c = np.zeros(self.V + 1)
        init_cell = self.V // 2
        for v in range(self.V):
            if v != init_cell:
                c[v + 1] = 1.0 / (self.V - 1)
        self.c = cp.Constant(c)

        # Parameters
        self.A = cp.Parameter((self.dxi, self.dk), name="A")
        self.b = cp.Parameter(self.dxi, name="b")
        self.D = cp.Parameter((self.T, self.V), name="D")

        # Decision variables (all <= 2D)
        self.pi  = cp.Variable((self.N * self.T, self.V), name="pi")
        self.nu  = cp.Variable((self.N * self.T, self.V), name="nu")
        self.lmb = cp.Variable((self.N, self.T), name="lambda")
        self.mu  = cp.Variable((self.N, self.V + 1), name="mu")
        self.Xi  = cp.Variable((self.N * self.V, self.dxi * (self.V + 1)), nonneg=True, name="Xi")

        self.problem = None
        self.layer = None

    def _get_pi(self, n, t):
        return self.pi[n * self.T + t, :]

    def _get_nu(self, n, t):
        return self.nu[n * self.T + t, :]

    def _get_xi(self, n, v):
        idx = n * self.V + v
        return cp.reshape(self.Xi[idx, :], (self.dxi, self.V + 1))

    def build(self):
        cons = []
        S, E, C = self.S, self.E, self.C
        AT = cp.transpose(self.A)

        # Nonnegativity (engineering stability)
        cons += [self.mu >= 0]
        cons += [self.pi >= 0]
        cons += [self.nu >= 0]
        cons += [self.lmb >= 0]
        # Xi is nonneg at creation

        # (a) distance-coupled ineq (D as parameter, DPP-safe)
        for n in range(self.N):
            cons += [cp.multiply(self.D[0, :], self._get_nu(n, 0)) <= 0]
            for t in range(1, self.T):
                cons += [cp.multiply(self.D[t, :], self._get_nu(n, t)) - self._get_pi(n, t) <= 0]

        # (b) support-function dualization aggregated over t (xi has no t-dim)
        for n in range(self.N):
            pi_sum = 0
            for t in range(self.T):
                pi_sum = pi_sum + self._get_pi(n, t)
            for v in range(self.V):
                rhs = pi_sum[v] * C
                cons += [AT @ self._get_xi(n, v) == rhs]

        # (c) main inequality: base - b^T Xi_sum >= 0
        b_col = cp.reshape(self.b, (self.dxi, 1))
        for n in range(self.N):
            bTXi_sum = 0
            for v in range(self.V):
                bTXi_sum = bTXi_sum + cp.transpose(b_col) @ self._get_xi(n, v)  # (1, V+1)
            for t in range(self.T):
                nu_row = cp.reshape(self._get_nu(n, t), (1, self.V))
                nu_term = nu_row @ S                                 # (1, V+1)
                lmb_term = self.lmb[n, t] * E                        # (1, V+1)
                base = self.Gamma[t] + nu_term + lmb_term
                if t == 0:
                    base = base + cp.reshape(self.mu[n, :], (1, self.V + 1))
                cons += [base - bTXi_sum >= 0]

        # ===== Objective (fix signs to avoid unboundedness) =====
        # minimize  + c^T mu  + sum lambda  + sum (b^T Xi)  + tiny L2 regularization
        obj = 0
        for n in range(self.N):
            obj = obj + ( self.c.T @ self.mu[n, :] )            # + c^T mu
            obj = obj + ( cp.sum(self.lmb[n, :]) )              # + sum lambda
            for v in range(self.V):
                obj = obj + ( cp.sum(cp.transpose(cp.reshape(self.b, (1, self.dxi))) @ self._get_xi(n, v)) )

        # small regularization for numerical stability
        obj = obj + 1e-8 * (cp.sum_squares(self.mu) + cp.sum_squares(self.lmb) + cp.sum_squares(self.Xi))

        self.problem = cp.Problem(cp.Minimize(obj), cons)

    def build_layer(self):
        self.layer = CvxpyLayer(
            self.problem,
            parameters=[self.A, self.b, self.D],
            variables=[self.mu, self.lmb, self.Xi]
        )

    @staticmethod
    def _map_delta_action(delta_action: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
        B = delta_action.shape[0]
        res = torch.full((B,), PureDualPersuitOptimization.ACTION_FOUND,
                         dtype=torch.long, device=delta_action.device)
        dx = delta_action[:, 0]
        dy = delta_action[:, 1]
        ax = torch.abs(dx)
        ay = torch.abs(dy)
        found = (ax <= eps) & (ay <= eps)
        forward = (~found) & (ay >= ax) & (dy > eps)
        res[forward] = PureDualPersuitOptimization.ACTION_FORWARD
        turn_right = (~found) & (ax > ay) & (dx > eps)
        res[turn_right] = PureDualPersuitOptimization.ACTION_TURN_RIGHT
        turn_left = (~found) & (ax > ay) & (dx < -eps)
        res[turn_left] = PureDualPersuitOptimization.ACTION_TURN_LEFT
        back_region = (~found) & (ay >= ax) & (dy < -eps)
        res[back_region & (dx >= 0)] = PureDualPersuitOptimization.ACTION_TURN_RIGHT
        res[back_region & (dx <  0)] = PureDualPersuitOptimization.ACTION_TURN_LEFT
        return res.unsqueeze(-1)

    def loss_from_solution(self, mu_val: torch.Tensor, lmb_val: torch.Tensor,
                           Xi_val: torch.Tensor, b_val: torch.Tensor) -> torch.Tensor:
        """
        Rebuild dual objective as torch scalar loss (mean over batch).
        This mirrors the objective signs in build().
        """
        B = mu_val.shape[0]
        device = mu_val.device
        dtype  = mu_val.dtype

        # + c^T mu
        c_t = torch.tensor(self.c.value, device=device, dtype=dtype).view(1, 1, -1)
        term1 =  (c_t * mu_val).sum(dim=-1).sum(dim=-1)     # (B,)

        # + sum lambda
        term2 =  lmb_val.sum(dim=(-1, -2))                  # (B,)

        if b_val.dim() == 1:
            b_val = b_val.unsqueeze(0).expand(B, -1)

        # + sum (b^T Xi)
        Xi_reshaped = Xi_val.view(B, self.N, self.V, self.dxi, self.V + 1)
        b_t = b_val.view(B, 1, 1, self.dxi, 1)
        bTXi = (b_t * Xi_reshaped).sum(dim=-2).squeeze(-2)  # (B, N, V, V+1)
        term3 =  bTXi.sum(dim=(1, 2, 3))                    # (B,)

        loss = (term1 + term2 + term3).mean()
        return loss

    def run(self, A_val: torch.Tensor, b_val: torch.Tensor, D_val: torch.Tensor, p_val: torch.Tensor):
        B = self.batch_size
        if A_val.dim() == 2: A_val = A_val.unsqueeze(0).expand(B, -1, -1)
        if b_val.dim() == 1: b_val = b_val.unsqueeze(0).expand(B, -1)
        if D_val.dim() == 2: D_val = D_val.unsqueeze(0).expand(B, -1, -1)
        if p_val.dim() == 2: p_val = p_val.unsqueeze(0).expand(B, -1, -1)
        mu_val, lmb_val, Xi_val = self.layer(A_val, b_val, D_val)
        delta = p_val[:, 1, :] - p_val[:, 0, :]
        mapped_values = self._map_delta_action(delta)
        loss = self.loss_from_solution(mu_val, lmb_val, Xi_val, b_val)
        return mapped_values, loss
