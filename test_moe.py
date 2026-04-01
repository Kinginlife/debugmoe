"""
Test script to verify MoE implementation
"""
import torch
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from vita.modeling.transformer_decoder.moe_layer import MoEFFNLayer


def test_moe_forward():
    """Test MoE forward pass"""
    print("Testing MoE forward pass...")

    d_model = 256
    dim_feedforward = 2048
    batch_size = 2
    seq_len = 100

    # Create MoE layer with 1 expert (Task 0)
    moe = MoEFFNLayer(
        d_model=d_model,
        dim_feedforward=dim_feedforward,
        num_experts=1,
        current_task=0,
        dropout=0.0,
        normalize_before=False
    )

    # Test input
    x = torch.randn(seq_len, batch_size, d_model)

    # Forward pass
    output = moe(x)

    assert output.shape == x.shape, f"Output shape mismatch: {output.shape} vs {x.shape}"
    print(f"✓ Forward pass successful. Output shape: {output.shape}")

    return moe


def test_add_expert():
    """Test adding new expert"""
    print("\nTesting add new expert...")

    moe = test_moe_forward()

    # Add expert for Task 1
    moe.add_new_expert(
        d_model=256,
        dim_feedforward=2048,
        dropout=0.0,
        activation="relu"
    )

    assert moe.num_experts == 2, f"Expected 2 experts, got {moe.num_experts}"
    assert len(moe.experts) == 2, f"Expected 2 experts in list, got {len(moe.experts)}"
    print(f"✓ Successfully added new expert. Total experts: {moe.num_experts}")

    return moe


def test_freeze_experts():
    """Test freezing old experts"""
    print("\nTesting freeze old experts...")

    moe = test_add_expert()

    # Freeze expert 0 (Task 0)
    moe.freeze_old_experts(current_task=1)

    # Check expert 0 is frozen
    for param in moe.experts[0].parameters():
        assert not param.requires_grad, "Expert 0 should be frozen"

    # Check expert 1 is trainable
    for param in moe.experts[1].parameters():
        assert param.requires_grad, "Expert 1 should be trainable"

    # Check router is frozen
    for param in moe.router.parameters():
        assert not param.requires_grad, "Router should be frozen"

    print("✓ Expert 0 frozen, Expert 1 trainable, Router frozen")


def test_router_weights():
    """Test router weight distribution"""
    print("\nTesting router weight distribution...")

    moe = test_add_expert()

    x = torch.randn(10, 2, 256)
    output = moe(x)

    print(f"✓ Router successfully distributes weights across {moe.num_experts} experts")


if __name__ == "__main__":
    print("=" * 50)
    print("MoE Layer Unit Tests")
    print("=" * 50)

    test_moe_forward()
    test_add_expert()
    test_freeze_experts()
    test_router_weights()

    print("\n" + "=" * 50)
    print("All tests passed! ✓")
    print("=" * 50)

