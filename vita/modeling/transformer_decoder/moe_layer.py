import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


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
        num_experts=1,
        current_task=0,
        dropout=0.0,
        activation="relu",
        normalize_before=False
    ):
        super().__init__()

        self.d_model = d_model
        self.dim_feedforward = dim_feedforward
        self.num_experts = num_experts
        self.current_task = current_task
        self.normalize_before = normalize_before

        # Router: outputs weight for each expert
        self.router = nn.Linear(d_model, num_experts)

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

    def forward_post(self, tgt):
        """Post-norm forward pass"""
        # Router computes weights for each expert
        router_logits = self.router(tgt)  # [seq_len, batch, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)
        # ====== DEBUG 专用代码：强制所有权重分配给 Expert 0 ======
        # 把所有权重清零
        router_weights = torch.zeros_like(router_weights)
        # 强制 Expert 0 (Task 0的旧专家) 权重为 1.0
        router_weights[..., 0] = 1.0

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

        router_logits = self.router(tgt2)
        router_weights = F.softmax(router_logits, dim=-1)
        # ====== DEBUG 专用代码：强制所有权重分配给 Expert 0 ======
        # 把所有权重清零
        router_weights = torch.zeros_like(router_weights)
        # 强制 Expert 0 (Task 0的旧专家) 权重为 1.0
        router_weights[..., 0] = 1.0

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
        """Freeze old experts (0 to current_task-1) and router"""
        # Freeze old experts
        for i in range(current_task):
            if i < len(self.experts):
                for param in self.experts[i].parameters():
                    param.requires_grad = False

        # Freeze router
        for param in self.router.parameters():
            param.requires_grad = False

    def add_new_expert(self, d_model, dim_feedforward, dropout, activation):
        """Add a new expert for new task"""
        new_expert = self._create_expert(d_model, dim_feedforward, dropout, activation)
        self.experts.append(new_expert)

        # Expand router output dimension
        old_router = self.router
        self.router = nn.Linear(d_model, self.num_experts + 1)

        # Copy old weights
        with torch.no_grad():
            self.router.weight[:self.num_experts] = old_router.weight
            self.router.bias[:self.num_experts] = old_router.bias
            nn.init.zeros_(self.router.weight[self.num_experts])
            self.router.bias[self.num_experts] = -10.0

        self.num_experts += 1
        self.current_task += 1


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")

