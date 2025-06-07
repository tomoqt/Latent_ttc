#!/usr/bin/env python
"""
This script loads a pre-trained GPT model with hierarchical loops, collects its hidden state
representations across different hierarchical loop iterations, performs PCA
on these representations, and visualizes the 2D and 3D trajectories of tokens.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from sklearn.decomposition import PCA
import os
import pickle
import math
import re # For sanitizing filenames
import matplotlib.patches as mpatches # Added for custom legends

# Assuming model.py is in the same directory or accessible in PYTHONPATH
from model import GPTConfig, GPT

def sanitize_filename_part(name_part):
    """Sanitizes a string to be used as part of a filename."""
    # Remove any characters that are not alphanumeric, underscore, or hyphen
    name_part = re.sub(r'[^a-zA-Z0-9_\-]', '', name_part)
    # Limit length to avoid overly long filenames
    return name_part[:50]

def collect_hierarchical_representations(model, input_tensor, max_iterations=50):
    """
    Collect representations from the hierarchical model by running multiple forward passes
    with different K values sampled dynamically.
    
    Args:
        model: GPT model with hierarchical loop structure
        input_tensor: Input tensor of shape (1, seq_len)
        max_iterations: Maximum number of iterations to collect
    
    Returns:
        List of tensors, each of shape (seq_len, n_embd) representing hidden states
    """
    representations = []
    model.eval()
    
    with torch.no_grad():
        for iteration in range(max_iterations):
            # Sample new K values for this iteration
            model.sample_K_values(device=input_tensor.device)
            
            # Forward pass through the model
            # We need to manually extract representations from the final layer
            # Let's run the forward pass and capture the output just before ln_f
            
            # Run through embeddings and transformer
            device = input_tensor.device
            b, t = input_tensor.size()
            pos = torch.arange(0, t, dtype=torch.long, device=device)
            
            tok_emb = model.transformer.wte(input_tensor)
            pos_emb = model.transformer.wpe(pos)
            
            x = (tok_emb + pos_emb) * math.sqrt(model.config.n_embd)
            x = model.transformer.drop(x)
            
            # Forward through the hierarchical tree
            x = model._forward_tree(x, model.transformer.tree, model.sampled_K_tensor, 0)
            
            # x now has shape (batch_size, seq_len, n_embd)
            # We want (seq_len, n_embd) for consistency with old format
            representations.append(x.squeeze(0).cpu())  # Remove batch dimension
            
            if iteration % 10 == 0:
                print(f"Collected {iteration + 1}/{max_iterations} representations")
    
    print(f"Collected {len(representations)} hierarchical representations")
    return representations

def compute_pca_and_transform(loop_representations_list, n_components=2):
    """
    Compute PCA on loop representations.
    Args:
        loop_representations_list: List of tensors, where each tensor is (seq_len, hidden_dim)
                                   representing hidden states for all tokens at a specific loop/pass.
        n_components: Number of PCA components.
    Returns:
        A tuple of (pca_model, transformed_reps_list).
        pca_model: The fitted sklearn PCA model.
        transformed_reps_list: List of numpy arrays, each (seq_len, n_components),
                               representing PCA-transformed states for each loop/pass.
    """
    if not loop_representations_list:
        raise ValueError("Loop representations list is empty.")

    # Stack all representations: (num_loops, seq_len, hidden_dim)
    # Convert to numpy for sklearn
    all_reps_np = torch.stack(loop_representations_list).numpy()
    num_loops, seq_len, hidden_dim = all_reps_np.shape

    # Reshape for PCA: (num_loops * seq_len, hidden_dim)
    data_for_pca = all_reps_np.reshape(-1, hidden_dim)

    # Ensure n_components is not more than available features or samples
    actual_n_components = min(n_components, data_for_pca.shape[0], data_for_pca.shape[1])
    if actual_n_components < n_components:
        print(f"Warning: Requested {n_components} PCA components, but only {actual_n_components} are feasible. Using {actual_n_components}.")

    # Fit PCA
    pca = PCA(n_components=actual_n_components)
    transformed_flat = pca.fit_transform(data_for_pca)

    # If actual_n_components is less than requested, pad with zeros for consistent shape if needed by downstream, though plotting will adapt.
    if actual_n_components < n_components:
        padding = np.zeros((transformed_flat.shape[0], n_components - actual_n_components))
        transformed_flat = np.hstack((transformed_flat, padding))

    # Reshape back to (num_loops, seq_len, n_components)
    transformed_reshaped = transformed_flat.reshape(num_loops, seq_len, n_components)

    # Convert to list of arrays, each (seq_len, n_components)
    transformed_reps_list_out = [transformed_reshaped[i] for i in range(num_loops)]
    
    print(f"PCA explained variance ratio for {actual_n_components} components: {pca.explained_variance_ratio_}")
    print(f"Total explained variance by {actual_n_components} components: {np.sum(pca.explained_variance_ratio_):.4f}")

    return pca, transformed_reps_list_out

PALETTES = [plt.cm.viridis, plt.cm.plasma, plt.cm.inferno, plt.cm.magma, 
            plt.cm.cividis, plt.cm.coolwarm, plt.cm.spring, plt.cm.autumn, 
            plt.cm.winter, plt.cm.summer]

def plot_pca_trajectories_2d(pca_transformed_reps_list, prompt_tokens_str, output_file_path, model_config=None):
    """
    Plot 2D PCA trajectories of token representations across hierarchical loops (all tokens on one plot).
    Args:
        pca_transformed_reps_list: List of numpy arrays, each (seq_len, 2),
                                   representing 2D PCA components for each iteration.
        prompt_tokens_str: List of strings, the decoded tokens of the prompt.
        output_file_path: Path to save the plot.
        model_config: Model configuration, used for hierarchical structure information.
    """
    if not pca_transformed_reps_list:
        print("No PCA results to plot for 2D combined trajectory.")
        return

    num_iterations = len(pca_transformed_reps_list)
    if num_iterations == 0 or pca_transformed_reps_list[0].shape[1] < 2:
        print("PCA results list for 2D combined plot is empty or has < 2 components.")
        return
    seq_len = pca_transformed_reps_list[0].shape[0]

    # Get hierarchical structure info
    hierarchical_info = "No hierarchical structure"
    if model_config and hasattr(model_config, 'hierarchical_spec') and model_config.hierarchical_spec:
        hierarchical_info = str(model_config.hierarchical_spec)
    elif model_config and hasattr(model_config, 'order1_groups') and model_config.order1_groups:
        hierarchical_info = str(model_config.order1_groups)
    
    max_loops = getattr(model_config, 'max_loops', num_iterations) if model_config else num_iterations

    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    token_cmap = plt.cm.get_cmap('tab10', seq_len if seq_len <= 10 else 20)
    
    for token_idx in range(seq_len):
        trajectory = np.array([pca_transformed_reps_list[iter_idx][token_idx, :2] for iter_idx in range(num_iterations)])
        token_label = prompt_tokens_str[token_idx] if token_idx < len(prompt_tokens_str) else f"Token {token_idx}"
        
        # Plot main trajectory line (colored by token)
        ax.plot(trajectory[:, 0], trajectory[:, 1], linestyle='-', 
                 color=token_cmap(token_idx % token_cmap.N), 
                 label=f"'{token_label}' (pos {token_idx})",
                 alpha=0.6, zorder=1)

        # Plot markers for each iteration
        for iter_idx in range(num_iterations):
            point_coords = trajectory[iter_idx, :]
            
            # Use a simple color progression for hierarchical iterations
            color_intensity = iter_idx / max(1, num_iterations - 1)
            point_color = plt.cm.viridis(color_intensity)

            ax.scatter(point_coords[0], point_coords[1], color=point_color, 
                       marker='o', s=20, alpha=0.8, zorder=2)

        if num_iterations > 0:
            # Start and end markers
            ax.scatter(trajectory[0, 0], trajectory[0, 1], s=50, 
                        color=token_cmap(token_idx % token_cmap.N), ec='black', marker='X', zorder=5)
            if num_iterations > 1:
                 ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=50, 
                             color=token_cmap(token_idx % token_cmap.N), ec='black', marker='*', zorder=5)

    title_info = f'Hierarchical 2D PCA Trajectories ({num_iterations} Iterations)'
    config_info = f'Max Loops: {max_loops}, Structure: {hierarchical_info}'
    plt.title(f'{title_info}\n{config_info}', fontsize=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    handles, labels = ax.get_legend_handles_labels()
    # Remove duplicates
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels: 
            unique_labels[label] = handle
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(output_file_path, dpi=300)
    plt.close()
    print(f"Hierarchical 2D PCA trajectory plot saved to {output_file_path}")

def plot_pca_trajectories_3d(pca_transformed_reps_list, prompt_tokens_str, output_file_path, model_config=None):
    """
    Plot 3D PCA trajectories of token representations across hierarchical loops (all tokens on one plot).
    Assumes pca_transformed_reps_list contains at least 3 components.
    """
    if not pca_transformed_reps_list:
        print("No PCA results to plot for 3D combined trajectory.")
        return
    num_iterations = len(pca_transformed_reps_list)
    if num_iterations == 0 or pca_transformed_reps_list[0].shape[1] < 3:
        print("PCA results list for 3D combined plot is empty or has < 3 components.")
        return
    seq_len = pca_transformed_reps_list[0].shape[0]

    # Get hierarchical structure info
    hierarchical_info = "No hierarchical structure"
    if model_config and hasattr(model_config, 'hierarchical_spec') and model_config.hierarchical_spec:
        hierarchical_info = str(model_config.hierarchical_spec)
    elif model_config and hasattr(model_config, 'order1_groups') and model_config.order1_groups:
        hierarchical_info = str(model_config.order1_groups)
    
    max_loops = getattr(model_config, 'max_loops', num_iterations) if model_config else num_iterations

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    token_cmap = plt.cm.get_cmap('tab10', seq_len if seq_len <= 10 else 20)

    for token_idx in range(seq_len):
        trajectory_3d = np.array([pca_transformed_reps_list[iter_idx][token_idx, :3] for iter_idx in range(num_iterations)])
        token_label = prompt_tokens_str[token_idx] if token_idx < len(prompt_tokens_str) else f"Token {token_idx}"
        
        ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], linestyle='-',
                color=token_cmap(token_idx % token_cmap.N), 
                label=f"'{token_label}' (pos {token_idx})", alpha=0.6, zorder=1)

        for iter_idx in range(num_iterations):
            point_coords = trajectory_3d[iter_idx, :]
            
            # Use a simple color progression for hierarchical iterations
            color_intensity = iter_idx / max(1, num_iterations - 1)
            point_color = plt.cm.viridis(color_intensity)
            
            ax.scatter(point_coords[0], point_coords[1], point_coords[2], color=point_color, 
                       marker='o', s=15, alpha=0.8, zorder=2, depthshade=True)

        if num_iterations > 0:
            ax.scatter(trajectory_3d[0, 0], trajectory_3d[0, 1], trajectory_3d[0, 2], s=30, 
                       color=token_cmap(token_idx % token_cmap.N), ec='black', marker='X', depthshade=True)
            if num_iterations > 1:
                ax.scatter(trajectory_3d[-1, 0], trajectory_3d[-1, 1], trajectory_3d[-1, 2], s=30, 
                           color=token_cmap(token_idx % token_cmap.N), ec='black', marker='*', depthshade=True)

    title_info = f'Hierarchical 3D PCA Trajectories ({num_iterations} Iterations)'
    config_info = f'Max Loops: {max_loops}, Structure: {hierarchical_info}'
    ax.set_title(f'{title_info}\n{config_info}', fontsize=10)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels: 
            unique_labels[label] = handle
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize='small')

    fig.tight_layout(rect=[0, 0, 0.85, 1])
    plt.savefig(output_file_path, dpi=300)
    plt.close(fig)
    print(f"Hierarchical 3D PCA trajectory plot saved to {output_file_path}")

def plot_single_token_pca_trajectory(pca_transformed_reps_list, token_idx_to_plot, token_str_label, output_file_path, 
                                     is_3d_plot=False, is_zoomed_view=False, num_last_steps_to_zoom=15, model_config=None):
    """
    Plot 2D or 3D PCA trajectory for a single token across hierarchical iterations.
    Can also produce a zoomed view of the last N steps.
    """
    if not pca_transformed_reps_list:
        print(f"No PCA results for token {token_str_label} (idx {token_idx_to_plot}).")
        return

    total_iterations = len(pca_transformed_reps_list)
    if total_iterations == 0:
        print(f"PCA list empty for token {token_str_label}.")
        return

    num_components_available = pca_transformed_reps_list[0].shape[1]
    if is_3d_plot and num_components_available < 3:
        print(f"Cannot make 3D plot for token {token_str_label}, only {num_components_available} PCA components available. Skipping 3D plot.")
        return
    if not is_3d_plot and num_components_available < 2:
        print(f"Cannot make 2D plot for token {token_str_label}, only {num_components_available} PCA components available. Skipping 2D plot.")
        return

    seq_len = pca_transformed_reps_list[0].shape[0]
    if token_idx_to_plot >= seq_len:
        print(f"Token index {token_idx_to_plot} OOB.")
        return

    full_trajectory = np.array([pca_transformed_reps_list[iter_idx][token_idx_to_plot, :] for iter_idx in range(total_iterations)])

    plot_title_suffix = f"across {total_iterations} Iterations"
    current_trajectory_to_plot = full_trajectory
    iter_indices_for_plot = np.arange(total_iterations)

    if is_zoomed_view:
        if total_iterations <= num_last_steps_to_zoom:
            print(f"Not enough iterations ({total_iterations}) to zoom for token '{token_str_label}'. Plotting full trajectory.")
        else:
            start_idx = total_iterations - num_last_steps_to_zoom
            current_trajectory_to_plot = full_trajectory[start_idx:, :]
            iter_indices_for_plot = np.arange(start_idx, total_iterations)
            plot_title_suffix = f"(Iterations {start_idx}-{total_iterations-1})"

    fig = plt.figure(figsize=(12, 9) if is_3d_plot else (10,8))
    ax = fig.add_subplot(111, projection='3d') if is_3d_plot else fig.add_subplot(111)
    
    # Get hierarchical structure info
    hierarchical_info = "No hierarchical structure"
    if model_config and hasattr(model_config, 'hierarchical_spec') and model_config.hierarchical_spec:
        hierarchical_info = str(model_config.hierarchical_spec)
    elif model_config and hasattr(model_config, 'order1_groups') and model_config.order1_groups:
        hierarchical_info = str(model_config.order1_groups)
    
    max_loops = getattr(model_config, 'max_loops', total_iterations) if model_config else total_iterations

    pc_data_to_plot = current_trajectory_to_plot[:, :3] if is_3d_plot else current_trajectory_to_plot[:, :2]

    # Plot connecting line
    if is_3d_plot:
        ax.plot(pc_data_to_plot[:, 0], pc_data_to_plot[:, 1], pc_data_to_plot[:, 2], linestyle='-', color='grey', alpha=0.5, zorder=1)
    else:
        ax.plot(pc_data_to_plot[:, 0], pc_data_to_plot[:, 1], linestyle='-', color='grey', alpha=0.6, zorder=1)

    # Plot markers with color progression
    for i, iter_idx_val in enumerate(iter_indices_for_plot):
        point = pc_data_to_plot[i, :]
        color_intensity = iter_idx_val / max(1, total_iterations - 1)
        point_color = plt.cm.viridis(color_intensity)
        
        scatter_kwargs = {
            "s": 60 if not is_3d_plot else 50, 
            "color": point_color,
            "ec": 'black', 
            "marker": 'o', 
            "zorder": 2,
            "alpha": 0.8
        }
        if is_3d_plot:
            scatter_kwargs["depthshade"] = True
            ax.scatter(point[0], point[1], point[2], **scatter_kwargs)
        else:
            ax.scatter(point[0], point[1], **scatter_kwargs)
    
    # Text annotations for iteration indices
    for i, iter_idx_val in enumerate(iter_indices_for_plot):
        point = pc_data_to_plot[i, :]
        if is_3d_plot:
            ax.text(point[0], point[1], point[2], f"{iter_idx_val}", size=7, zorder=4, color='k')
        else:
            ax.annotate(f"{iter_idx_val}", (point[0], point[1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, zorder=4)
    
    # Start and End markers
    if pc_data_to_plot.shape[0] > 0:
        first_iter_idx = iter_indices_for_plot[0]
        color_intensity_start = first_iter_idx / max(1, total_iterations - 1)
        start_color = plt.cm.viridis(color_intensity_start)

        if is_3d_plot:
            ax.scatter(pc_data_to_plot[0, 0], pc_data_to_plot[0, 1], pc_data_to_plot[0, 2], 
                       s=80, color=start_color, ec='black', marker='X', zorder=3, 
                       label=f'Start: Iteration {first_iter_idx}', depthshade=True)
        else:
            ax.scatter(pc_data_to_plot[0, 0], pc_data_to_plot[0, 1], 
                       s=100, color=start_color, ec='black', marker='X', zorder=3, 
                       label=f'Start: Iteration {first_iter_idx}')

        if pc_data_to_plot.shape[0] > 1:
            last_iter_idx = iter_indices_for_plot[-1]
            color_intensity_end = last_iter_idx / max(1, total_iterations - 1)
            end_color = plt.cm.viridis(color_intensity_end)
            
            if is_3d_plot:
                ax.scatter(pc_data_to_plot[-1, 0], pc_data_to_plot[-1, 1], pc_data_to_plot[-1, 2], 
                           s=80, color=end_color, ec='black', marker='P', zorder=3, 
                           label=f'End: Iteration {last_iter_idx}', depthshade=True)
            else:
                ax.scatter(pc_data_to_plot[-1, 0], pc_data_to_plot[-1, 1], 
                           s=100, color=end_color, ec='black', marker='P', zorder=3, 
                           label=f'End: Iteration {last_iter_idx}')

    dim_str = "3D" if is_3d_plot else "2D"
    base_title = f'{dim_str} Hierarchical PCA Trajectory for Token: "{token_str_label}" (pos {token_idx_to_plot}) {plot_title_suffix}'
    config_info = f'Max Loops: {max_loops}, Structure: {hierarchical_info}'
    ax.set_title(f'{base_title}\n{config_info}', fontsize=10)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    if is_3d_plot:
        ax.set_zlabel('Principal Component 3')

    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), fontsize='small', loc='best')

    ax.grid(True, linestyle='--', alpha=0.7)
    if not is_3d_plot:
        ax.axhline(0, color='black', linewidth=0.5, alpha=0.5)
        ax.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    if is_zoomed_view and pc_data_to_plot.shape[0] > 1:
        x_min, x_max = pc_data_to_plot[:, 0].min(), pc_data_to_plot[:, 0].max()
        y_min, y_max = pc_data_to_plot[:, 1].min(), pc_data_to_plot[:, 1].max()
        x_margin = (x_max - x_min) * 0.1 if (x_max - x_min) > 1e-5 else 0.1
        y_margin = (y_max - y_min) * 0.1 if (y_max - y_min) > 1e-5 else 0.1
        ax.set_xlim(x_min - x_margin, x_max + x_margin)
        ax.set_ylim(y_min - y_margin, y_max + y_margin)
        if is_3d_plot:
            z_min, z_max = pc_data_to_plot[:, 2].min(), pc_data_to_plot[:, 2].max()
            z_margin = (z_max - z_min) * 0.1 if (z_max - z_min) > 1e-5 else 0.1
            ax.set_zlim(z_min - z_margin, z_max + z_margin)
    
    fig.tight_layout()
    fig.savefig(output_file_path, dpi=300)
    plt.close(fig)
    view_type = "Zoomed" if is_zoomed_view else "Full"
    print(f"{view_type} individual {dim_str} hierarchical PCA trajectory for token \"{token_str_label}\" saved to {output_file_path}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize hierarchical loop representations from a GPT model using PCA.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Full path to the model checkpoint (.pt file)')
    parser.add_argument('--prompt', type=str, default="Hello world, this is a test.", help='Input prompt string')
    parser.add_argument('--output_dir', type=str, default='hierarchical_analysis_output', help='Directory to save plots')
    parser.add_argument('--meta_path', type=str, default='data/fineweb/meta.pkl', help='Path to meta.pkl for tokenizer')
    parser.add_argument('--max_loops_override', type=int, default=None, help='Override model config max_loops')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--n_pca_components', type=int, default=2, help='Number of PCA components (2 or 3 for plotting)')
    parser.add_argument('--max_iterations', type=int, default=10, help='Number of iterations to collect representations')
    parser.add_argument('--num_last_steps_for_zoom', type=int, default=15, help='Number of last iterations for zoomed plots')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu', weights_only=False)
    if 'model_args' not in checkpoint:
        print("Error: 'model_args' not found in checkpoint."); return
    
    # Store model_args to pass to plotting functions
    gpt_model_config = checkpoint['model_args']
    
    # Override max_loops if requested
    if args.max_loops_override is not None:
        gpt_model_config['loop_max_loops'] = args.max_loops_override
        gpt_model_config['max_loops'] = args.max_loops_override
    
    # Ensure hierarchical_spec exists for plotting
    if 'hierarchical_spec' not in gpt_model_config:
        if 'order1_groups' in gpt_model_config:
            gpt_model_config['hierarchical_spec'] = gpt_model_config['order1_groups']
        else:
            gpt_model_config['hierarchical_spec'] = []

    gptconf = GPTConfig(**gpt_model_config)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix): 
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    
    print(f"Model: vocab {model.config.vocab_size}, block {model.config.block_size}, n_layer {model.config.n_layer}")
    if hasattr(model.config, 'hierarchical_spec') and model.config.hierarchical_spec:
        print(f"Hierarchical structure: {model.config.hierarchical_spec}")
    if hasattr(model.config, 'loop_cfg') and model.config.loop_cfg:
        print(f"Max loops per group: {model.config.loop_cfg.max_loops}")

    print(f"Loading tokenizer from {args.meta_path}...")
    tokenizer_encode_fn = None
    tokenizer_decode_fn_for_single_id_to_str = None

    try:
        with open(args.meta_path, 'rb') as f:
            meta = pickle.load(f)

        if 'encode' in meta and callable(meta['encode']) and \
           'decode' in meta and callable(meta['decode']):
            print("Using encode/decode methods from meta.pkl (expected for BPE/SentencePiece).")
            tokenizer_encode_fn = meta['encode']
            tokenizer_decode_fn_for_single_id_to_str = lambda token_id: meta['decode']([token_id])
        else:
            print("Warning: meta.pkl does not provide .encode/.decode methods. "
                  "Attempting to use 'stoi' and 'itos' from meta.pkl. "
                  "This is likely character-level tokenization if 'encode'/'decode' are missing.")
            if 'stoi' not in meta or 'itos' not in meta:
                print(f"Error: meta.pkl is missing 'stoi'/'itos' and also "
                      "lacks .encode/.decode methods. Cannot proceed with tokenization.")
                return
            
            stoi, itos = meta['stoi'], meta['itos']
            tokenizer_encode_fn = lambda s: [stoi[c] for c in s if c in stoi]
            tokenizer_decode_fn_for_single_id_to_str = lambda token_id: itos.get(token_id, '?')

    except FileNotFoundError:
        print(f"Error: Tokenizer meta file not found at {args.meta_path}"); return
    except pickle.UnpicklingError:
        print(f"Error: Could not unpickle tokenizer meta file from {args.meta_path}"); return
    except Exception as e:
        print(f"Error loading or initializing tokenizer from {args.meta_path}: {e}"); return

    if not tokenizer_encode_fn or not tokenizer_decode_fn_for_single_id_to_str:
        print("Tokenizer functions not initialized. Exiting.")
        return

    print(f"Tokenizing prompt: \"{args.prompt}\"")
    input_ids = tokenizer_encode_fn(args.prompt)

    if not input_ids:
        print("Error: Could not tokenize prompt (resulted in empty ID list)."); return
    
    if len(input_ids) > model.config.block_size:
        input_ids = input_ids[:model.config.block_size]
        print(f"Prompt truncated to {len(input_ids)} tokens to fit model block size {model.config.block_size}.")
    
    # Generate string representations for each token ID
    prompt_tokens_str = [tokenizer_decode_fn_for_single_id_to_str(id_) for id_ in input_ids]
    print(f"Token IDs: {input_ids}, Strings: {prompt_tokens_str}")

    # Convert IDs to tensor for the model
    input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    print(f"Collecting hierarchical representations across {args.max_iterations} iterations...")
    hierarchical_representations = collect_hierarchical_representations(
        model, input_tensor, max_iterations=args.max_iterations)

    if not hierarchical_representations:
        print("Error: No hierarchical representations returned."); return
    
    print(f"Collected {len(hierarchical_representations)} sets of representations. Shape of first: {hierarchical_representations[0].shape if hierarchical_representations else 'N/A'}")
    prompt_seq_len = hierarchical_representations[0].shape[0] if hierarchical_representations else 0
    if prompt_seq_len == 0: 
        print("Error: Zero sequence length from representations."); return

    print(f"Computing PCA with up to {args.n_pca_components} components...")
    try:
        pca_model, transformed_reps_list = compute_pca_and_transform(hierarchical_representations, n_components=args.n_pca_components)
    except ValueError as e: 
        print(f"Error during PCA: {e}"); return
    if not transformed_reps_list: 
        print("PCA resulted in empty list."); return

    # --- Plotting ---
    # 2D Plots
    if args.n_pca_components >= 2:
        reps_for_plotting_2d = [arr[:, :2] for arr in transformed_reps_list]
        combined_plot_filename_2d = f"hierarchical_pca_trajectories_2D_pc{min(args.n_pca_components,2)}.png"
        combined_plot_filepath_2d = os.path.join(args.output_dir, combined_plot_filename_2d)
        print(f"Plotting combined 2D hierarchical PCA trajectories to {combined_plot_filepath_2d}...")
        plot_pca_trajectories_2d(reps_for_plotting_2d, prompt_tokens_str, combined_plot_filepath_2d, model_config=gptconf)
    else:
        print("Skipping 2D plots as n_pca_components < 2.")

    # 3D Plots
    if args.n_pca_components >= 3:
        reps_for_plotting_3d = [arr[:, :3] for arr in transformed_reps_list]
        combined_plot_filename_3d = f"hierarchical_pca_trajectories_3D_pc{min(args.n_pca_components,3)}.png"
        combined_plot_filepath_3d = os.path.join(args.output_dir, combined_plot_filename_3d)
        print(f"Plotting combined 3D hierarchical PCA trajectories to {combined_plot_filepath_3d}...")
        plot_pca_trajectories_3d(reps_for_plotting_3d, prompt_tokens_str, combined_plot_filepath_3d, model_config=gptconf)
    else:
        print("Skipping 3D plots as n_pca_components < 3.")

    individual_plots_dir = os.path.join(args.output_dir, "individual_token_plots")
    os.makedirs(individual_plots_dir, exist_ok=True)
    print(f"Plotting individual token hierarchical PCA trajectories to {individual_plots_dir}...")

    for token_idx in range(prompt_seq_len):
        token_str = prompt_tokens_str[token_idx] if token_idx < len(prompt_tokens_str) else f"UNK_{token_idx}"
        sanitized_token_str = sanitize_filename_part(token_str if token_str != '?' else f"UNK_{token_idx}")
        num_available_iterations = len(transformed_reps_list)

        # Individual 2D plots (full and zoomed)
        if args.n_pca_components >=2:
            filename_2d_full = f"token_{token_idx}_{sanitized_token_str}_hierarchical_pca_2D_full.png"
            filepath_2d_full = os.path.join(individual_plots_dir, filename_2d_full)
            plot_single_token_pca_trajectory(transformed_reps_list, token_idx, token_str, filepath_2d_full, 
                                             is_3d_plot=False, is_zoomed_view=False, model_config=gptconf)
            if num_available_iterations > args.num_last_steps_for_zoom:
                filename_2d_zoomed = f"token_{token_idx}_{sanitized_token_str}_hierarchical_pca_2D_zoomed_last{args.num_last_steps_for_zoom}.png"
                filepath_2d_zoomed = os.path.join(individual_plots_dir, filename_2d_zoomed)
                plot_single_token_pca_trajectory(transformed_reps_list, token_idx, token_str, filepath_2d_zoomed, 
                                                 is_3d_plot=False, is_zoomed_view=True, num_last_steps_to_zoom=args.num_last_steps_for_zoom, model_config=gptconf)
        
        # Individual 3D plots (full and zoomed)
        if args.n_pca_components >=3:
            filename_3d_full = f"token_{token_idx}_{sanitized_token_str}_hierarchical_pca_3D_full.png"
            filepath_3d_full = os.path.join(individual_plots_dir, filename_3d_full)
            plot_single_token_pca_trajectory(transformed_reps_list, token_idx, token_str, filepath_3d_full, 
                                             is_3d_plot=True, is_zoomed_view=False, model_config=gptconf)
            if num_available_iterations > args.num_last_steps_for_zoom:
                filename_3d_zoomed = f"token_{token_idx}_{sanitized_token_str}_hierarchical_pca_3D_zoomed_last{args.num_last_steps_for_zoom}.png"
                filepath_3d_zoomed = os.path.join(individual_plots_dir, filename_3d_zoomed)
                plot_single_token_pca_trajectory(transformed_reps_list, token_idx, token_str, filepath_3d_zoomed, 
                                                 is_3d_plot=True, is_zoomed_view=True, num_last_steps_to_zoom=args.num_last_steps_for_zoom, model_config=gptconf)

    print("Hierarchical analysis complete.")

if __name__ == '__main__':
    main() 