"""
Test script for the new hierarchical loop system.
"""

import torch
from model import GPT, GPTConfig, LoopConfig

def test_hierarchical_looping():
    """Test the hierarchical looping system with the specified configuration."""
    print("Testing hierarchical looping system...")
    
    # Create loop configuration
    loop_cfg = LoopConfig(
        max_loops=5,
        noise_scale=0.1,
        concat_init=True,
        auto_exit=False
    )
    
    # Test the hierarchical specification: [[[2], [3]], [[4], [5]]]
    # This means:
    # - Group 1: Process layer 2, then layer 3, repeat this sequence up to 5 times
    # - Group 2: Process layer 4, then layer 5, repeat this sequence up to 5 times
    config = GPTConfig(
        block_size=64,
        vocab_size=1000,
        n_layer=6,
        n_head=4,
        n_embd=128,
        hierarchical_spec=[[[2], [3]], [[4], [5]]],  # Two order-2 groups
        loop_cfg=loop_cfg
    )
    
    # Create model
    model = GPT(config)
    print(f"Model created with {model.get_num_params()/1e6:.2f}M parameters")
    print(f"Effective n_layer: {model.effective_n_layer}")
    print(f"Hierarchical spec: {config.hierarchical_spec}")
    
    # Test forward pass
    batch_size = 2
    seq_len = 32
    idx = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    targets = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    print("\nTesting forward pass...")
    logits, loss = model(idx, targets)
    print(f"Forward pass successful. Logits shape: {logits.shape}, Loss: {loss.item():.4f}")
    
    # Test generation
    print("\nTesting generation...")
    model.eval()
    with torch.no_grad():
        generated = model.generate(idx[:1, :10], max_new_tokens=20)
    print(f"Generation successful. Output shape: {generated.shape}")
    
    # Print the tree structure
    print("\nModel tree structure:")
    print_tree_structure(model.tree, indent=0)
    
    print("\nHierarchical looping test passed!")

def print_tree_structure(group, indent=0):
    """Print the hierarchical tree structure."""
    prefix = "  " * indent
    print(f"{prefix}{group.name} (order {group.order}, max_loops={group.loop_cfg.max_loops})")
    
    for child in group.child_modules:
        if hasattr(child, 'order'):  # It's a HierarchicalGroup
            print_tree_structure(child, indent + 1)
        else:  # It's a Block
            print(f"{prefix}  Block")

def test_simple_case():
    """Test with a simpler hierarchical case."""
    print("\n" + "="*50)
    print("Testing simple hierarchical case...")
    
    loop_cfg = LoopConfig(max_loops=2, noise_scale=0.0, concat_init=False)
    
    # Simple case: [[[0], [1]]] - one order-2 group containing two order-1 groups
    config = GPTConfig(
        block_size=32,
        vocab_size=100,
        n_layer=3,
        n_head=2,
        n_embd=64,
        hierarchical_spec=[[[0], [1]]],  # One order-2 group
        loop_cfg=loop_cfg
    )
    
    model = GPT(config)
    idx = torch.randint(0, config.vocab_size, (1, 16))
    
    with torch.no_grad():
        logits, _ = model(idx)
    
    print(f"Simple case passed. Output shape: {logits.shape}")
    print("Tree structure:")
    print_tree_structure(model.tree, indent=0)

if __name__ == "__main__":
    print("Testing new hierarchical loop implementation...\n")
    
    test_hierarchical_looping()
    test_simple_case()
    
    print("\nAll hierarchical tests completed successfully!") 