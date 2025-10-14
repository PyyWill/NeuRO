from torch import nn, Tensor
import torch


class PICNN(nn.Module):
    """
    Partially input-convex neural network mapping from input example
    x of shape [input_dim] and y of shape [y_dim] to a scalar score output
    s(x, y)
    """

    def __init__(self, input_dim = 32, y_dim = 768, hidden_dim = 256, n_layers = 3,
                 y_in_output_layer = True, gamma = 0. ,
                 output_dim = 81,):
        super().__init__()
        self.input_dim = input_dim
        self.y_dim = y_dim
        self.hidden_dim = hidden_dim
        L = n_layers
        self.L = L

        # bag of tricks for feasibility
        self.gamma = gamma
        self.y_in_output_layer = y_in_output_layer

        null_module = nn.Module()

        self.W_hat_layers = nn.ModuleList(
            [null_module]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(1, L+1)]
        )
        self.W_bar_layers = nn.ModuleList(
            [null_module]
            + [nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(1, L)]
            + [nn.Linear(hidden_dim, output_dim, bias=False)]
        )
        self.V_hat_layers = nn.ModuleList(
            [nn.Linear(input_dim, y_dim)]
            + [nn.Linear(hidden_dim, y_dim) for _ in range(1, L+1)]
        )
        self.V_bar_layers = nn.ModuleList(
            [nn.Linear(y_dim, hidden_dim, bias=False) for _ in range(L)]
            + [nn.Linear(y_dim, 1, bias=False)]
        )
        self.b_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(1, L)]
            + [nn.Linear(hidden_dim, output_dim)]
        )
        self.u_layers = nn.ModuleList(
            [nn.Linear(input_dim, hidden_dim)]
            + [nn.Linear(hidden_dim, hidden_dim) for _ in range(1, L)]  # Used to be L+1
        )
        self.clamp_weights()

    def clamp_weights(self) -> None:
        """Clamps weights of all the W_bar layers to be ≥ 0."""
        with torch.no_grad():
            for layer in self.W_bar_layers:
                if isinstance(layer, nn.Linear):
                    layer.weight.clamp_(min=0)
            if not self.y_in_output_layer:
                self.V_bar_layers[-1].weight.fill_(0.)

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """Computes the score for the given input examples (x, y).

        Args:
            x: shape [batch_size, input_dim], input
            y: shape [batch_size, y_dim], labels

        Returns:
            s: shape [batch_size], score
        """
        ReLU = nn.ReLU()
        u = x
        # Initialize sigma to participate in computational graph
        sigma = torch.zeros(x.shape[0], self.hidden_dim, device=x.device, dtype=x.dtype, requires_grad=True)
        
        for l in range(self.L):
            if l > 0:
                W_hat_vec = ReLU(self.W_hat_layers[l](u))
                W_sigma = self.W_bar_layers[l](W_hat_vec * sigma)
            else:
                W_sigma = 0
            V_hat_vec = self.V_hat_layers[l](u)        # shape [batch, d]
            V_y = self.V_bar_layers[l](V_hat_vec * y)
            b = self.b_layers[l](u)
            sigma = ReLU(W_sigma + V_y + b)
            u = ReLU(self.u_layers[l](u))
            
        l = self.L
        W_hat_vec = ReLU(self.W_hat_layers[l](u))
        W_sigma = self.W_bar_layers[l](W_hat_vec * sigma)

        V_y = 0.
        if self.y_in_output_layer:
            V_hat_vec = self.V_hat_layers[l](u)
            V_y = self.V_bar_layers[l](V_hat_vec * y)
        b = self.b_layers[l](u)

        if self.gamma == 0.:
            output = W_sigma + V_y + b
        else:
            kappa = self.gamma * torch.norm(y, p=float('inf'), dim=1)
            kappa = kappa.unsqueeze(-1)
            output = W_sigma + V_y + kappa + b

        # For robust optimization, return scalar score
        result = output[..., 0]  # [batch_size]
        return result

    def predict_uncertainty_params(self, x, q_threshold, Ld=0, num_anchors=10):
        """
        Predict uncertainty set parameters A(θ) and b(θ,q) from PICNN.
        
        Args:
            x: Input tensor of shape [batch_size, input_dim]
            q_threshold: Quantile threshold for uncertainty set
            Ld: Lifting dimension (default 0 for minimal LP)
            num_anchors: Number of anchor points for polyhedral approximation
            
        Returns:
            A_theta: Tensor of shape [batch_size, d_xi, d_k]
            b_theta_q: Tensor of shape [batch_size, d_xi]
        """
        assert torch.is_grad_enabled(), "PICNN anchors require grad; wrap call in torch.enable_grad()"
        B = x.shape[0]
        V = 3 * 3  # Grid size (3x3 grid to match dual layer edge=3)
        d_k = V + Ld
        d_xi = 2 * Ld + 1
        
        M = min(num_anchors, d_xi)
        
        A_theta = torch.zeros(B, d_xi, d_k, device=x.device, dtype=x.dtype)
        b_theta_q = torch.zeros(B, d_xi, device=x.device, dtype=x.dtype)
        
        A_rows = []
        b_rows = []
        
        for m in range(M):
            y_m = torch.softmax(torch.randn(B, d_k, device=x.device, dtype=x.dtype), dim=-1)
            y_m = y_m.detach().clone().requires_grad_(True)
            
            g_m = self.forward(x, y_m)
            
            grad_y_m = torch.autograd.grad(
                outputs=g_m,
                inputs=y_m,
                grad_outputs=torch.ones_like(g_m),
                retain_graph=True, 
                create_graph=False,
                allow_unused=False
            )[0]
            
            support = (grad_y_m * y_m).sum(dim=-1, keepdim=True)
            if not torch.is_tensor(q_threshold):
                q_threshold = torch.tensor(q_threshold, device=g_m.device, dtype=g_m.dtype)
            q_threshold = q_threshold.expand(B).view(B, 1)
            b_row = q_threshold - (g_m.view(B, 1) - support)
            
            A_rows.append(grad_y_m)
            b_rows.append(b_row.squeeze(-1))
        
        if len(A_rows) > 0:
            A_theta[:, :M, :] = torch.stack(A_rows, dim=1)
            b_theta_q[:, :M] = torch.stack(b_rows, dim=1)
        
        return A_theta, b_theta_q

    def _generate_anchor_points(self, V, Ld, num_anchors, device, dtype):
        """Generate anchor points for polyhedral approximation."""
        d_k = V + Ld
        
        # Generate random anchor points
        anchor_points = torch.randn(num_anchors, d_k, device=device, dtype=dtype)
        
        # Normalize to ensure they are valid probability distributions
        if Ld == 0:  # No lifting, ensure probabilities sum to 1
            anchor_points = torch.softmax(anchor_points, dim=1)
        else:  # With lifting, just normalize
            anchor_points = anchor_points / torch.norm(anchor_points, dim=1, keepdim=True)
        
        return anchor_points


