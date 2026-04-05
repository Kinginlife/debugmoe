import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFNLayer(nn.Module):
    """Mixture of Experts FFN Layer for Incremental Learning

    Args:
        d_model: input dimension
        dim_feedforward: FFN hidden dimension
        num_experts: total number of experts for current task
        current_task: current task ID (0-indexed)
        dropout: dropout rate
        normalize_before: whether to use pre-norm
        activation: activation function name
    """

    def __init__(
        self,
        d_model,
        dim_feedforward=2048,
        num_experts=2,
        current_task=0,
        dropout=0.0,
        activation="relu",
        normalize_before=False,
        router_svd_energy=0.99,
    ):
        super().__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.current_task = current_task
        self.normalize_before = normalize_before
        self.router_svd_energy = router_svd_energy

        # Base router: historical dimensions (frozen in incremental phase)
        self.router = nn.Linear(d_model, num_experts)
        # New router head for the current incremental task (1-dim, no bias)
        self.new_router = None
        # Right singular vectors of old router row-space: shape [d_model, r]
        self.register_buffer("old_router_basis", None, persistent=False)

        # Experts: each expert is an independent FFN
        self.experts = nn.ModuleList([
            self._create_expert(d_model, dim_feedforward, dropout, activation)
            for _ in range(num_experts)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = _get_activation_fn(activation)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _create_expert(self, d_model, dim_feedforward, dropout, activation):
        """Create a single expert (standard FFN)"""
        return nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )

    def _compute_router_logits(self, x):
        """Compute router logits, concatenating temporary new router if present."""
        logits = self.router(x)
        if self.new_router is not None:
            new_logits = self.new_router(x)
            logits = torch.cat([logits, new_logits], dim=-1)
        return logits

    def _compute_top2_router_weights(self, x):
        """Compute Top-2 sparse router weights over experts.

        Returns:
            router_weights: same shape as logits, only top-2 entries are non-zero and renormalized.
        """
        router_logits = self._compute_router_logits(x)
        k = min(2, router_logits.shape[-1])

        # Select top-k experts per token
        topk_logits, topk_indices = torch.topk(router_logits, k=k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        # Scatter sparse top-k weights back to full expert dimension
        router_weights = torch.zeros_like(router_logits)
        router_weights.scatter_(-1, topk_indices, topk_weights.to(router_weights.dtype))
        return router_weights

    def forward_post(self, tgt):
        """Post-norm forward pass"""
        # Top-2 sparse routing weights for each token
        router_weights = self._compute_top2_router_weights(tgt)  # [seq_len, batch, num_experts]

        # Weighted combination of all experts
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            # Apply activation to first layer output
            x = expert[0](tgt)  # linear1
            x = self.activation(x)
            x = expert[1](x)    # dropout
            x = expert[2](x)    # linear2

            weighted_out = x * router_weights[..., i:i+1]
            expert_outputs.append(weighted_out)

        tgt2 = sum(expert_outputs)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        """Pre-norm forward pass"""
        tgt2 = self.norm(tgt)

        router_weights = self._compute_top2_router_weights(tgt2)

        expert_outputs = []
        for i, expert in enumerate(self.experts):
            x = expert[0](tgt2)
            x = self.activation(x)
            x = expert[1](x)
            x = expert[2](x)

            weighted_out = x * router_weights[..., i:i+1]
            expert_outputs.append(weighted_out)

        tgt2 = sum(expert_outputs)
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        """Forward pass with residual connection and normalization"""
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

    def freeze_old_experts(self, current_task):
        """Freeze old experts (0 to current_task-1) and base router."""
        # Freeze old experts
        for i in range(current_task):
            if i < len(self.experts):
                for param in self.experts[i].parameters():
                    param.requires_grad = False

        # Freeze historical router dimensions
        for param in self.router.parameters():
            param.requires_grad = False
            param.grad = None

        # Keep newly added router dimension trainable
        if self.new_router is not None:
            for param in self.new_router.parameters():
                param.requires_grad = True

    def _compute_old_router_basis(self, energy_ratio=0.99):
        """Compute right singular vectors spanning old router row-space."""
        with torch.no_grad():
            w = self.router.weight.detach()  # [num_old_experts, d_model]
            if w.numel() == 0:
                self.old_router_basis = None
                return

            # Full SVD on old router rows
            # w = U S Vh, row-space basis in columns of V = Vh^T
            _, s, vh = torch.linalg.svd(w, full_matrices=False)
            if s.numel() == 0:
                self.old_router_basis = None
                return

            s2 = s.pow(2)
            total = torch.clamp(s2.sum(), min=1e-12)
            cum = torch.cumsum(s2, dim=0) / total
            r = int(torch.searchsorted(cum, torch.tensor(energy_ratio, device=cum.device)).item()) + 1
            r = max(1, min(r, vh.shape[0]))

            basis = vh[:r, :].transpose(0, 1).contiguous()  # [d_model, r]
            self.old_router_basis = basis.to(self.router.weight.device)

    def _project_to_old_router_orthogonal(self, vec):
        """Project row vector(s) to orthogonal complement of old router row-space."""
        if self.old_router_basis is None:
            return vec
        b = self.old_router_basis  # [d_model, r]
        # vec: [..., d_model]
        return vec - (vec @ b) @ b.transpose(0, 1)

    def _register_new_router_weight_grad_projection_hook(self):
        """Hard gradient projection for new_router.weight onto orthogonal complement."""
        if self.new_router is None:
            return

        def _hook(grad):
            # grad shape: [1, d_model]
            if grad is None or self.old_router_basis is None:
                return grad
            return self._project_to_old_router_orthogonal(grad)

        self.new_router.weight.register_hook(_hook)

    def add_new_expert(self, d_model, dim_feedforward, dropout, activation):
        """Add a new expert and a temporary 1-dim router head for new task."""
        new_expert = self._create_expert(d_model, dim_feedforward, dropout, activation)
        self.experts.append(new_expert)

        # 1) Build old router basis via SVD (before adding new dimension)
        self._compute_old_router_basis(energy_ratio=self.router_svd_energy)

        # 2) Create temporary new router dimension (trainable only for current task, no bias)
        self.new_router = nn.Linear(d_model, 1, bias=False)
        self.new_router = self.new_router.to(self.router.weight.device)

        # 3) Orthogonal initialization in feature space (project random vector to orthogonal complement)
        with torch.no_grad():
            nn.init.normal_(self.new_router.weight, mean=0.0, std=1.0)
            w = self.new_router.weight.data  # [1, d_model]
            w = self._project_to_old_router_orthogonal(w)

            # normalize + small scale to avoid initial routing collapse
            norm = torch.clamp(w.norm(p=2, dim=1, keepdim=True), min=1e-12)
            w = w / norm
            old_row_norm_mean = torch.clamp(self.router.weight.data.norm(p=2, dim=1).mean(), min=1e-6)
            scale = 0.05 * old_row_norm_mean
            self.new_router.weight.copy_(w * scale)

        # 4) Register hard gradient projection hook
        self._register_new_router_weight_grad_projection_hook()

        self.num_experts += 1
        self.current_task += 1

    def fix_router(self):
        """Merge temporary new_router into router and clear new_router."""
        if self.new_router is None:
            return

        trained_router = nn.Linear(self.d_model, self.router.out_features + 1).to(self.router.weight.device)

        with torch.no_grad():
            trained_router.weight[:-1].copy_(self.router.weight.data)
            trained_router.bias[:-1].copy_(self.router.bias.data)
            trained_router.weight[-1:].copy_(self.new_router.weight.data)
            # new_router has no bias; initialize merged bias row to 0
            trained_router.bias[-1:].zero_()

        self.router = trained_router
        self.new_router = None
        self.old_router_basis = None


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")
