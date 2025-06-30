#!/usr/bin/env python
"""
This script loads a pre-trained GPT model, collects its hidden state
representations across different loops/layer group passes, performs PCA
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

def box_counting_dimension(points):
    """
    Estimates the box-counting (Minkowski-Bouligand) dimension.
    This is often used as an estimate for the Hausdorff dimension.

    Args:
        points (np.ndarray): An array of shape (n_points, n_dims).
        
    Returns:
        float: The estimated box-counting dimension.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0

    # Calculate the bounding box of the point cloud
    # This gives us the minimum and maximum coordinates in each dimension
    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    
    # Edge case: if all points are identical (degenerate case), dimension is 0
    # This happens when the trajectory doesn't actually move through space
    if np.all(min_coords == max_coords):
        return 0.0

    # Normalize the point cloud to unit cube [0,1]^d
    # This makes the analysis scale-invariant and easier to work with
    side_lengths = max_coords - min_coords
    # Add small epsilon to avoid division by zero
    points_normalized = (points - min_coords) / (side_lengths + 1e-9)

    # Create a range of scales (box sizes) to test
    # We use logarithmic spacing from 0.001 to 1.0 to cover multiple orders of magnitude
    # This range allows us to see how the point count changes across different resolutions
    scales = np.logspace(np.log10(0.001), np.log10(1.0), num=20, endpoint=False)
    
    # For each scale, count how many boxes are needed to cover the point cloud
    counts = []
    for scale in scales:
        # Skip invalid scales
        if scale <= 0: continue
        
        # Discretize the normalized points by dividing by scale and taking floor
        # This effectively creates a grid of boxes of size 'scale'
        discretized = np.floor(points_normalized / scale)
        
        # Count unique boxes (grid cells) that contain at least one point
        # This gives us N(ε) in the box-counting formula
        count = len(np.unique(discretized, axis=0))
        counts.append(count)
        
    # Convert to numpy array for vectorized operations
    counts = np.array(counts)
    
    # Filter out scales where we only have 1 box (degenerate case)
    # We need at least 2 different box counts to fit a line
    valid_indices = (counts > 1)
    if np.sum(valid_indices) < 2:
        return np.nan  # Not enough data to estimate dimension
        
    # Keep only the valid scales and counts for the linear fit
    scales = scales[valid_indices]
    counts = counts[valid_indices]
    
    # Fit a line to log(N(ε)) vs log(1/ε) using least squares
    # The slope of this line is the box-counting dimension
    # Formula: log(N(ε)) = D * log(1/ε) + c, where D is the dimension
    coeffs = np.polyfit(np.log(1/scales), np.log(counts), 1)
    
    # Return the slope (first coefficient), which is our dimension estimate
    return coeffs[0]

def sanitize_filename_part(name_part):
    """Sanitizes a string to be used as part of a filename."""
    # Remove any characters that are not alphanumeric, underscore, or hyphen
    name_part = re.sub(r'[^a-zA-Z0-9_\-]', '', name_part)
    # Limit length to avoid overly long filenames
    return name_part[:50]

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
    Plot 2D PCA trajectories of token representations across loops (all tokens on one plot).
    Args:
        pca_transformed_reps_list: List of numpy arrays, each (seq_len, 2),
                                   representing 2D PCA components for each loop/pass.
        prompt_tokens_str: List of strings, the decoded tokens of the prompt.
        output_file_path: Path to save the plot.
        model_config: Model configuration, used for loop group information.
    """
    if not pca_transformed_reps_list:
        print("No PCA results to plot for 2D combined trajectory.")
        return

    num_loops = len(pca_transformed_reps_list)
    if num_loops == 0 or pca_transformed_reps_list[0].shape[1] < 2:
        print("PCA results list for 2D combined plot is empty or has < 2 components.")
        return
    seq_len = pca_transformed_reps_list[0].shape[0]

    num_loop_groups = 1
    if model_config and hasattr(model_config, 'loop_groups') and model_config.loop_groups:
        num_loop_groups = len(model_config.loop_groups)
    
    total_model_loops = model_config.max_loops if model_config and hasattr(model_config, 'max_loops') else num_loops

    if model_config and hasattr(model_config, 'loop_groups') and model_config.loop_groups:
        loop_group_str = str(model_config.loop_groups)
    else:
        loop_group_str = "Sequential (1 group)"

    plt.figure(figsize=(14, 10))
    ax = plt.gca()
    
    token_cmap = plt.cm.get_cmap('tab10', seq_len if seq_len <= 10 else 20)
    
    for token_idx in range(seq_len):
        trajectory = np.array([pca_transformed_reps_list[loop_idx][token_idx, :2] for loop_idx in range(num_loops)])
        token_label = prompt_tokens_str[token_idx] if token_idx < len(prompt_tokens_str) else f"Token {token_idx}"
        
        # Plot main trajectory line (colored by token)
        ax.plot(trajectory[:, 0], trajectory[:, 1], linestyle='-', 
                 color=token_cmap(token_idx % token_cmap.N), 
                 label=f"'{token_label}' (pos {token_idx})" if num_loop_groups == 1 else None, # Avoid duplicate labels if markers are colored
                 alpha=0.4, zorder=1)

        # Plot markers colored by loop group and progression
        for loop_k in range(num_loops):
            point_coords = trajectory[loop_k, :]
            original_loop_idx = loop_k

            group_idx = original_loop_idx % num_loop_groups
            cmap_for_point = PALETTES[group_idx % len(PALETTES)]
            occurrence = original_loop_idx // num_loop_groups
            # Max occurrence index for this group_idx
            max_occ_idx = (total_model_loops - 1 - group_idx) // num_loop_groups
            
            norm_val = occurrence / max(1, max_occ_idx) if max_occ_idx >= 0 and max_occ_idx > 0 else 0.0
            point_color = cmap_for_point(norm_val)

            ax.scatter(point_coords[0], point_coords[1], color=point_color, 
                       marker='o', s=20, alpha=0.8, zorder=2, 
                       label=f"'{token_label}' (pos {token_idx})" if loop_k ==0 and num_loop_groups > 1 else None)


        if num_loops > 0:
            # Start and end markers (overall trajectory for this token)
            ax.scatter(trajectory[0, 0], trajectory[0, 1], s=50, 
                        color=token_cmap(token_idx % token_cmap.N), ec='black', marker='X', zorder=5,
                        label=f"Start '{token_label}'" if num_loop_groups == 1 and token_idx == 0 else None) # Simplified legend
            if num_loops > 1:
                 ax.scatter(trajectory[-1, 0], trajectory[-1, 1], s=50, 
                             color=token_cmap(token_idx % token_cmap.N), ec='black', marker='*', zorder=5,
                             label=f"End '{token_label}'" if num_loop_groups == 1 and token_idx == 0 else None)


    title_info = f'Combined 2D PCA Trajectories ({num_loops} Data Loops)'
    config_info = f'Model: {total_model_loops} Loops, Groups: {num_loop_groups}, Structure: {loop_group_str}'
    plt.title(f'{title_info}\n{config_info}', fontsize=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    handles, labels = ax.get_legend_handles_labels()
    # Filter out duplicate labels for tokens if num_loop_groups > 1
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels : unique_labels[label] = handle
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)

    if num_loop_groups > 1:
        palette_legend_handles = []
        # Only add handles for groups that are actually present if num_loops < num_loop_groups
        actual_groups_in_plot = sorted(list(set(idx % num_loop_groups for idx in range(num_loops))))
        for grp_idx in actual_groups_in_plot:
            cmap = PALETTES[grp_idx % len(PALETTES)]
            patch = mpatches.Patch(color=cmap(0.6), label=f'Group {grp_idx}') # cmap(0.6) for representative color
            palette_legend_handles.append(patch)
        if palette_legend_handles:
            from matplotlib.legend import Legend # Import Legend here
            palette_labels = [h.get_label() for h in palette_legend_handles]
            leg2 = Legend(ax, palette_legend_handles, palette_labels, title="Loop Groups", loc='lower right', fontsize='x-small', bbox_to_anchor=(1, 0))
            ax.add_artist(leg2)
            plt.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust rect for two legends
    else:
        plt.tight_layout(rect=[0, 0, 0.85, 1])

    plt.savefig(output_file_path, dpi=300)
    plt.close()
    print(f"Combined 2D PCA trajectory plot saved to {output_file_path}")

def plot_pca_trajectories_3d(pca_transformed_reps_list, prompt_tokens_str, output_file_path, model_config=None):
    """
    Plot 3D PCA trajectories of token representations across loops (all tokens on one plot).
    Assumes pca_transformed_reps_list contains at least 3 components.
    """
    if not pca_transformed_reps_list:
        print("No PCA results to plot for 3D combined trajectory.")
        return
    num_loops = len(pca_transformed_reps_list)
    if num_loops == 0 or pca_transformed_reps_list[0].shape[1] < 3:
        print("PCA results list for 3D combined plot is empty or has < 3 components.")
        return
    seq_len = pca_transformed_reps_list[0].shape[0]

    num_loop_groups = 1
    if model_config and hasattr(model_config, 'loop_groups') and model_config.loop_groups:
        num_loop_groups = len(model_config.loop_groups)
    total_model_loops = model_config.max_loops if model_config and hasattr(model_config, 'max_loops') else num_loops

    if model_config and hasattr(model_config, 'loop_groups') and model_config.loop_groups:
        loop_group_str = str(model_config.loop_groups)
    else:
        loop_group_str = "Sequential (1 group)"

    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    token_cmap = plt.cm.get_cmap('tab10', seq_len if seq_len <= 10 else 20)

    for token_idx in range(seq_len):
        trajectory_3d = np.array([pca_transformed_reps_list[loop_idx][token_idx, :3] for loop_idx in range(num_loops)]) # Use first 3 components
        token_label = prompt_tokens_str[token_idx] if token_idx < len(prompt_tokens_str) else f"Token {token_idx}"
        
        ax.plot(trajectory_3d[:, 0], trajectory_3d[:, 1], trajectory_3d[:, 2], linestyle='-',
                color=token_cmap(token_idx % token_cmap.N), 
                label=f"'{token_label}' (pos {token_idx})" if num_loop_groups == 1 else None, alpha=0.4, zorder=1)

        for loop_k in range(num_loops):
            point_coords = trajectory_3d[loop_k, :]
            original_loop_idx = loop_k

            group_idx = original_loop_idx % num_loop_groups
            cmap_for_point = PALETTES[group_idx % len(PALETTES)]
            occurrence = original_loop_idx // num_loop_groups
            max_occ_idx = (total_model_loops - 1 - group_idx) // num_loop_groups
            
            norm_val = occurrence / max(1, max_occ_idx) if max_occ_idx >= 0 and max_occ_idx > 0 else 0.0
            point_color = cmap_for_point(norm_val)
            
            ax.scatter(point_coords[0], point_coords[1], point_coords[2], color=point_color, 
                       marker='o', s=15, alpha=0.8, zorder=2, depthshade=True,
                       label=f"'{token_label}' (pos {token_idx})" if loop_k == 0 and num_loop_groups > 1 else None)


        if num_loops > 0:
            ax.scatter(trajectory_3d[0, 0], trajectory_3d[0, 1], trajectory_3d[0, 2], s=30, 
                       color=token_cmap(token_idx % token_cmap.N), ec='black', marker='X', depthshade=True,
                       label=f"Start '{token_label}'" if num_loop_groups == 1 and token_idx == 0 else None)
            if num_loops > 1:
                ax.scatter(trajectory_3d[-1, 0], trajectory_3d[-1, 1], trajectory_3d[-1, 2], s=30, 
                           color=token_cmap(token_idx % token_cmap.N), ec='black', marker='*', depthshade=True,
                           label=f"End '{token_label}'" if num_loop_groups == 1 and token_idx == 0 else None)

    title_info = f'Combined 3D PCA Trajectories ({num_loops} Data Loops)'
    config_info = f'Model: {total_model_loops} Loops, Groups: {num_loop_groups}, Structure: {loop_group_str}'
    ax.set_title(f'{title_info}\n{config_info}', fontsize=10)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels : unique_labels[label] = handle
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1.1, 0.5), fontsize='small')

    if num_loop_groups > 1:
        palette_legend_handles = []
        actual_groups_in_plot = sorted(list(set(idx % num_loop_groups for idx in range(num_loops))))
        for grp_idx in actual_groups_in_plot:
            cmap = PALETTES[grp_idx % len(PALETTES)]
            patch = mpatches.Patch(color=cmap(0.6), label=f'Group {grp_idx}')
            palette_legend_handles.append(patch)
        if palette_legend_handles:
            from matplotlib.legend import Legend # Ensure Legend is imported if not globally
            palette_labels = [h.get_label() for h in palette_legend_handles]
            leg2 = Legend(ax, palette_legend_handles, palette_labels, title="Loop Groups", loc='lower right', fontsize='x-small', bbox_to_anchor=(1.1, 0))
            ax.add_artist(leg2)
            fig.tight_layout(rect=[0, 0, 0.80, 1]) # Adjust rect for two legends
    else:
         fig.tight_layout(rect=[0, 0, 0.85, 1])
         
    plt.savefig(output_file_path, dpi=300)
    plt.close(fig)
    print(f"Combined 3D PCA trajectory plot saved to {output_file_path}")

def plot_single_token_pca_trajectory(pca_transformed_reps_list, token_idx_to_plot, token_str_label, output_file_path, 
                                     is_3d_plot=False, is_zoomed_view=False, num_last_steps_to_zoom=15, model_config=None):
    """
    Plot 2D or 3D PCA trajectory for a single token across loops.
    Can also produce a zoomed view of the last N steps.
    """
    if not pca_transformed_reps_list:
        print(f"No PCA results for token {token_str_label} (idx {token_idx_to_plot}).")
        return

    total_data_loops = len(pca_transformed_reps_list) # Actual loops in data
    if total_data_loops == 0:
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

    full_trajectory = np.array([pca_transformed_reps_list[loop_idx][token_idx_to_plot, :] for loop_idx in range(total_data_loops)])

    plot_title_suffix = f"across {total_data_loops} Loops"
    current_trajectory_to_plot = full_trajectory
    loop_indices_for_plot = np.arange(total_data_loops) # These are original loop indices

    if is_zoomed_view:
        if total_data_loops <= num_last_steps_to_zoom:
            print(f"Not enough loops ({total_data_loops}) to zoom for token '{token_str_label}'. Plotting full trajectory.")
        else:
            start_idx = total_data_loops - num_last_steps_to_zoom
            current_trajectory_to_plot = full_trajectory[start_idx:, :]
            loop_indices_for_plot = np.arange(start_idx, total_data_loops)
            plot_title_suffix = f"(Loops {start_idx}-{total_data_loops-1})"

    fig = plt.figure(figsize=(12, 9) if is_3d_plot else (10,8))
    ax = fig.add_subplot(111, projection='3d') if is_3d_plot else fig.add_subplot(111)
    
    total_model_loops = total_data_loops 
    loop_group_str = "Sequential (1 group)"
    if model_config and hasattr(model_config, 'loop_groups') and model_config.loop_groups:
        num_loop_groups = len(model_config.loop_groups)
        loop_group_str = str(model_config.loop_groups)
    if model_config and hasattr(model_config, 'max_loops'):
        total_model_loops = model_config.max_loops

    pc_data_to_plot = current_trajectory_to_plot[:, :3] if is_3d_plot else current_trajectory_to_plot[:, :2]

    # Plot connecting line
    if is_3d_plot:
        ax.plot(pc_data_to_plot[:, 0], pc_data_to_plot[:, 1], pc_data_to_plot[:, 2], linestyle='-', color='grey', alpha=0.5, zorder=1)
    else:
        ax.plot(pc_data_to_plot[:, 0], pc_data_to_plot[:, 1], linestyle='-', color='grey', alpha=0.6, zorder=1)

    all_legend_handles = []

    # Plot markers for each group
    for group_iter in range(num_loop_groups):
        current_cmap = PALETTES[group_iter % len(PALETTES)]
        
        indices_for_group_in_view = [
            i for i, original_loop_idx in enumerate(loop_indices_for_plot)
            if (original_loop_idx % num_loop_groups) == group_iter
        ]

        if not indices_for_group_in_view:
            continue

        pc_data_group_subset = pc_data_to_plot[indices_for_group_in_view]
        original_loop_indices_of_subset = loop_indices_for_plot[indices_for_group_in_view]
        
        occurrences_in_group = original_loop_indices_of_subset // num_loop_groups
        max_occurrence_idx_for_group = (total_model_loops - 1 - group_iter) // num_loop_groups
        
        color_values_norm = occurrences_in_group / max(1, max_occurrence_idx_for_group) if max_occurrence_idx_for_group >=0 and max_occurrence_idx_for_group > 0 else np.zeros_like(occurrences_in_group)

        scatter_kwargs = {
            "s": 60 if not is_3d_plot else 50, "c": color_values_norm, "cmap": current_cmap,
            "ec": 'black', "marker": 'o', "zorder": 2,
            "label": f"Group {group_iter} Loops"
        }
        if is_3d_plot:
            scatter_kwargs["depthshade"] = True
            # Need to ensure pc_data_group_subset has 3 columns if is_3d_plot
            ax.scatter(pc_data_group_subset[:, 0], pc_data_group_subset[:, 1], pc_data_group_subset[:, 2], **scatter_kwargs)
        else:
            ax.scatter(pc_data_group_subset[:, 0], pc_data_group_subset[:, 1], **scatter_kwargs)
    
    # Text annotations for loop indices
    for i, loop_idx_val in enumerate(loop_indices_for_plot):
        point = pc_data_to_plot[i, :]
        if is_3d_plot:
            ax.text(point[0], point[1], point[2], f"{loop_idx_val}", size=7, zorder=4, color='k')
        else:
            ax.annotate(f"{loop_idx_val}", (point[0], point[1]), textcoords="offset points", xytext=(5,5), ha='center', fontsize=8, zorder=4)
    
    # Start and End markers, colored by their group's palette and progression
    if pc_data_to_plot.shape[0] > 0:
        # First point in view
        first_loop_in_view_original_idx = loop_indices_for_plot[0]
        first_group = first_loop_in_view_original_idx % num_loop_groups
        first_cmap = PALETTES[first_group % len(PALETTES)]
        first_occurrence = first_loop_in_view_original_idx // num_loop_groups
        first_max_occ_idx = (total_model_loops - 1 - first_group) // num_loop_groups
        first_norm_val = first_occurrence / max(1, first_max_occ_idx) if first_max_occ_idx >=0 and first_max_occ_idx > 0 else 0.0
        start_marker_color = first_cmap(first_norm_val)

        start_marker_label = f'Loop {first_loop_in_view_original_idx} (Group {first_group})'
        if is_3d_plot:
            ax.scatter(pc_data_to_plot[0, 0], pc_data_to_plot[0, 1], pc_data_to_plot[0, 2], 
                       s=80, color=start_marker_color, ec='black', marker='X', zorder=3, 
                       label=start_marker_label, depthshade=True)
        else:
            ax.scatter(pc_data_to_plot[0, 0], pc_data_to_plot[0, 1], 
                       s=100, color=start_marker_color, ec='black', marker='X', zorder=3, 
                       label=start_marker_label)

        if pc_data_to_plot.shape[0] > 1:
            # Last point in view
            last_loop_in_view_original_idx = loop_indices_for_plot[-1]
            last_group = last_loop_in_view_original_idx % num_loop_groups
            last_cmap = PALETTES[last_group % len(PALETTES)]
            last_occurrence = last_loop_in_view_original_idx // num_loop_groups
            last_max_occ_idx = (total_model_loops - 1 - last_group) // num_loop_groups
            last_norm_val = last_occurrence / max(1, last_max_occ_idx) if last_max_occ_idx >=0 and last_max_occ_idx > 0 else 0.0
            end_marker_color = last_cmap(last_norm_val)
            
            end_marker_label = f'Loop {last_loop_in_view_original_idx} (Group {last_group})'
            if is_3d_plot:
                ax.scatter(pc_data_to_plot[-1, 0], pc_data_to_plot[-1, 1], pc_data_to_plot[-1, 2], 
                           s=80, color=end_marker_color, ec='black', marker='P', zorder=3, 
                           label=end_marker_label, depthshade=True)
            else:
                ax.scatter(pc_data_to_plot[-1, 0], pc_data_to_plot[-1, 1], 
                           s=100, color=end_marker_color, ec='black', marker='P', zorder=3, 
                           label=end_marker_label)

    dim_str = "3D" if is_3d_plot else "2D"
    base_title = f'{dim_str} PCA Trajectory for Token: "{token_str_label}" (pos {token_idx_to_plot}) {plot_title_suffix}'
    config_info = f'Model: {total_model_loops} Loops, Groups: {num_loop_groups}, Structure: {loop_group_str}'
    ax.set_title(f'{base_title}\n{config_info}', fontsize=10)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    if is_3d_plot:
        ax.set_zlabel('Principal Component 3')

    # Update legend
    handles, labels = ax.get_legend_handles_labels()
    # Create a dictionary to keep the first occurrence of each label to ensure uniqueness
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
    print(f"{view_type} individual {dim_str} PCA trajectory for token \"{token_str_label}\" saved to {output_file_path}")

def plot_convergence_diagnostics(diagnostics_data, output_dir, model_config):
    """
    Plots the collected convergence diagnostics for each loop group.
    """
    if not diagnostics_data:
        print("No convergence diagnostics data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for group_key, metrics in diagnostics_data.items():
        num_iterations = len(metrics['delta_norm'])
        if num_iterations < 2:
            print(f"Not enough data to plot diagnostics for {group_key}.")
            continue
        
        iterations = np.arange(num_iterations)
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Convergence Diagnostics for {group_key.replace("_", " ")}', fontsize=16)

        # Plot Δ-norm
        axs[0, 0].plot(iterations, metrics['delta_norm'], marker='o', linestyle='-')
        axs[0, 0].set_title('Change in Representation Norm (Δ-norm)')
        axs[0, 0].set_xlabel('Loop Iteration')
        axs[0, 0].set_ylabel(r'$\|x_{k+1} - x_k\|_2$')
        axs[0, 0].grid(True)

        # Plot Δ-angle
        # Skip first value which is None
        axs[0, 1].plot(iterations[1:], metrics['delta_angle'][1:], marker='o', linestyle='-')
        axs[0, 1].set_title('Angle between Successive Changes (Δ-angle)')
        axs[0, 1].set_xlabel('Loop Iteration')
        axs[0, 1].set_ylabel(r'$\cos\angle(\Delta_k, \Delta_{k-1})$')
        axs[0, 1].grid(True)

        # Plot Hidden-vector norm
        axs[1, 0].plot(iterations, metrics['hidden_norm'], marker='o', linestyle='-')
        axs[1, 0].set_title('Hidden Vector Norm')
        axs[1, 0].set_xlabel('Loop Iteration')
        axs[1, 0].set_ylabel(r'$\|x_k\|$')
        axs[1, 0].grid(True)

        # Plot Logit drift
        # Skip first value which is None
        axs[1, 1].plot(iterations[1:], metrics['logit_drift'][1:], marker='o', linestyle='-')
        axs[1, 1].set_title('Logit Drift (KL Divergence)')
        axs[1, 1].set_xlabel('Loop Iteration')
        axs[1, 1].set_ylabel(r'KL$(p_{k-1}\|p_k)$')
        axs[1, 1].set_yscale('log')
        axs[1, 1].grid(True, which="both")
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"convergence_diagnostics_{group_key}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)
        print(f"Convergence diagnostics plot saved to {plot_filepath}")

def plot_jacobian_eigenvalues(eigenvalue_data, output_dir, model_config):
    """
    Plots the eigenvalue spectrum of the Jacobian for each loop group and token.
    """
    if not eigenvalue_data:
        print("No Jacobian eigenvalue data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for group_key, token_eigvals_list in eigenvalue_data.items():
        num_tokens = len(token_eigvals_list)
        if num_tokens == 0:
            continue

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, aspect='equal')
        
        # Unit circle
        unit_circle = mpatches.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.5, label='Unit Circle')
        ax.add_patch(unit_circle)
        
        cmap = plt.cm.get_cmap('viridis', num_tokens)
        
        for token_idx, eigvals in enumerate(token_eigvals_list):
            eigvals_complex = np.array(eigvals)
            ax.scatter(eigvals_complex.real, eigvals_complex.imag, s=10, 
                       color=cmap(token_idx / max(1, num_tokens - 1)), 
                       alpha=0.6, label=f'Token {token_idx}')

        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title(f'Jacobian Eigenvalue Spectrum for {group_key.replace("_", " ")}')
        ax.grid(True)
        ax.axhline(0, color='grey', lw=0.5)
        ax.axvline(0, color='grey', lw=0.5)
        
        # Set limits to be slightly larger than the unit circle for better visualization
        lim_max = 1.1
        all_eigvals = np.concatenate(token_eigvals_list)
        max_abs = np.max(np.abs(all_eigvals))
        if max_abs > 1.0:
            lim_max = max_abs * 1.1
            
        ax.set_xlim(-lim_max, lim_max)
        ax.set_ylim(-lim_max, lim_max)
        
        # To avoid overcrowding, only show legend for a few tokens
        handles, labels = ax.get_legend_handles_labels()
        if num_tokens > 10:
            # Show legend for first, middle, and last token
            indices_to_show = [0, num_tokens // 2, num_tokens - 1]
            handles = [handles[i] for i in indices_to_show]
            labels = [labels[i] for i in indices_to_show]
        ax.legend(handles, labels, title="Token Position", loc='best')

        plt.tight_layout()
        plot_filename = f"jacobian_eigvals_{group_key}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)
        print(f"Jacobian eigenvalues plot saved to {plot_filepath}")

def plot_global_diagnostics(diagnostics_data, output_dir):
    """
    Plots the collected global convergence diagnostics across all layers/steps.
    """
    if not diagnostics_data or not diagnostics_data['delta_norm']:
        print("No global diagnostics data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)
    num_steps = len(diagnostics_data['delta_norm'])
    steps = np.arange(num_steps)

    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Global Diagnostics Across Forward Pass', fontsize=16)

    # Plot Δ-norm
    axs[0, 0].plot(steps, diagnostics_data['delta_norm'], marker='o', linestyle='-', markersize=3)
    axs[0, 0].set_title('Change in Representation Norm (Δ-norm)')
    axs[0, 0].set_xlabel('Global Step (Layer or Loop Iteration)')
    axs[0, 0].set_ylabel(r'$\|x_{k+1} - x_k\|_2$')
    axs[0, 0].grid(True)

    # Plot Δ-angle
    axs[0, 1].plot(steps[1:], diagnostics_data['delta_angle'][1:], marker='o', linestyle='-', markersize=3)
    axs[0, 1].set_title('Angle between Successive Changes (Δ-angle)')
    axs[0, 1].set_xlabel('Global Step (Layer or Loop Iteration)')
    axs[0, 1].set_ylabel(r'$\cos\angle(\Delta_k, \Delta_{k-1})$')
    axs[0, 1].grid(True)

    # Plot Hidden-vector norm
    axs[1, 0].plot(steps, diagnostics_data['hidden_norm'], marker='o', linestyle='-', markersize=3)
    axs[1, 0].set_title('Hidden Vector Norm')
    axs[1, 0].set_xlabel('Global Step (Layer or Loop Iteration)')
    axs[1, 0].set_ylabel(r'$\|x_k\|$')
    axs[1, 0].grid(True)

    # Plot Logit drift
    axs[1, 1].plot(steps[1:], diagnostics_data['logit_drift'][1:], marker='o', linestyle='-', markersize=3)
    axs[1, 1].set_title('Logit Drift (KL Divergence)')
    axs[1, 1].set_xlabel('Global Step (Layer or Loop Iteration)')
    axs[1, 1].set_ylabel(r'KL$(p_{k-1}\|p_k)$')
    axs[1, 1].set_yscale('log')
    axs[1, 1].grid(True, which="both")
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filepath = os.path.join(output_dir, "global_diagnostics.png")
    plt.savefig(plot_filepath, dpi=300)
    plt.close(fig)
    print(f"Global diagnostics plot saved to {plot_filepath}")

def plot_jacobian_eigenvalue_trajectory(eigval_trajectory_data, output_dir):
    """
    Plots the trajectory of the max-magnitude Jacobian eigenvalue for each token in a loop group.
    """
    if not eigval_trajectory_data:
        print("No Jacobian eigenvalue trajectory data to plot.")
        return

    os.makedirs(output_dir, exist_ok=True)

    for group_key, token_trajectories in eigval_trajectory_data.items():
        num_tokens = len(token_trajectories)
        if num_tokens == 0 or len(token_trajectories[0]) == 0:
            continue
            
        fig, ax = plt.subplots(figsize=(12, 8))
        num_iterations = len(token_trajectories[0])
        iterations = np.arange(num_iterations)
        cmap = plt.cm.get_cmap('viridis', num_tokens)

        for token_idx, trajectory in enumerate(token_trajectories):
            ax.plot(iterations, trajectory, marker='.', linestyle='-', 
                    color=cmap(token_idx / max(1, num_tokens - 1)),
                    alpha=0.7, label=f'Token {token_idx}')

        ax.axhline(1.0, color='r', linestyle='--', label='Stability Boundary (|λ|=1)')
        ax.set_xlabel('Loop Iteration')
        ax.set_ylabel('Max Eigenvalue Magnitude |λ|')
        ax.set_title(f'Jacobian Max Eigenvalue Trajectory for {group_key.replace("_", " ")}')
        ax.grid(True)
        ax.legend(title="Token Position", loc='center left', bbox_to_anchor=(1, 0.5))
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plot_filename = f"jacobian_eigval_trajectory_{group_key}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)
        print(f"Jacobian eigenvalue trajectory plot saved to {plot_filepath}")

def plot_max_singular_values(model, output_dir):
    """
    Computes and plots the maximum singular value for each 2D weight matrix in the model.
    """
    max_singular_values = {}
    for name, param in model.named_parameters():
        # We are interested in 2D weight matrices
        if param.dim() == 2:
            with torch.no_grad():
                try:
                    # More efficient to just get singular values.
                    # Move to CPU and ensure float32 for SVD robustness.
                    S = torch.linalg.svdvals(param.to('cpu', dtype=torch.float32))
                    max_sv = S.max().item()
                    max_singular_values[name] = max_sv
                except torch.linalg.LinAlgError as e:
                    print(f"SVD computation failed for parameter {name}: {e}. Skipping.")
                except Exception as e:
                    print(f"An unexpected error occurred during SVD for {name}: {e}. Skipping.")

    if not max_singular_values:
        print("No 2D weight matrices found for which to plot singular values.")
        return

    # Sorting by name for a consistent plot order
    sorted_names = sorted(max_singular_values.keys())
    sorted_values = [max_singular_values[name] for name in sorted_names]

    plt.figure(figsize=(15, 12))
    plt.bar(range(len(sorted_values)), sorted_values)
    plt.xticks(range(len(sorted_names)), sorted_names, rotation=90, fontsize='small')
    plt.ylabel('Maximum Singular Value')
    plt.title('Maximum Singular Values of Model Weight Matrices')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filepath = os.path.join(output_dir, "max_singular_values.png")
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Maximum singular values plot saved to {plot_filepath}")

def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize loop representations from a GPT model using PCA.")
    parser.add_argument('--checkpoint_path', type=str, required=True, help='Full path to the model checkpoint (.pt file)')
    parser.add_argument('--prompt', type=str, default="Hello world, this is a test.", help='Input prompt string')
    parser.add_argument('--output_dir', type=str, default='representation_analysis_output', help='Directory to save plots')
    parser.add_argument('--meta_path', type=str, default='data/fineweb/meta.pkl', help='Path to meta.pkl for tokenizer')
    parser.add_argument('--max_loops_override', type=int, default=None, help='Override model config max_loops')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--n_pca_components', type=int, default=2, help='Number of PCA components (2 or 3 for plotting)')
    parser.add_argument('--max_new_tokens_for_analysis', type=int, default=1, help='Number of new tokens for representation collection trigger')
    parser.add_argument('--num_last_steps_for_zoom', type=int, default=15, help='Number of last loops for zoomed plots')
    parser.add_argument('--calculate_hausdorff_dimension', action='store_true', help='If set, calculate the Hausdorff dimension of trajectories.')
    parser.add_argument('--track_convergence_diagnostics', action='store_true', help='If set, track and plot convergence diagnostics.')
    parser.add_argument('--calculate_jacobian', action='store_true', help='If set, calculate and plot Jacobian eigenvalues.')
    parser.add_argument('--calculate_jacobian_trajectory', action='store_true', help='If set, plot the trajectory of the max Jacobian eigenvalue.')
    parser.add_argument('--track_global_diagnostics', action='store_true', help='If set, plot diagnostics across the entire model forward pass.')
    parser.add_argument('--plot_singular_values', action='store_true', help='If set, plot the max singular values of the model\'s weight matrices.')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name for logging analysis.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name for logging analysis. If not provided, a new one will be generated.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    wandb_logging_enabled = args.wandb_project and args.wandb_run_name
    if wandb_logging_enabled:
        import wandb
        # Resume the run if it exists, otherwise create a new one.
        # This allows logging analysis artifacts to the same run as training.
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, id=args.wandb_run_name, resume="allow")

    print(f"Loading checkpoint from {args.checkpoint_path}...")
    checkpoint = torch.load(args.checkpoint_path, map_location='cpu')
    if 'model_args' not in checkpoint:
        print("Error: 'model_args' not found in checkpoint."); return
    
    # Store model_args to pass to plotting functions
    gpt_model_config = checkpoint['model_args']
    # Convert dict to a class-like object if it's a dict, for attribute access like model_config.loop_groups
    # Or ensure gptconf = GPTConfig(**gpt_model_config) is the source of truth
    
    gpt_model_config['loops_representation'] = True
    gpt_model_config['automatic_loop_exit'] = False
    if args.track_convergence_diagnostics:
        gpt_model_config['track_convergence_diagnostics'] = True
    if args.calculate_jacobian:
        gpt_model_config['calculate_jacobian'] = True
    if args.calculate_jacobian_trajectory:
        gpt_model_config['calculate_jacobian_trajectory'] = True
    if args.track_global_diagnostics:
        gpt_model_config['track_global_diagnostics'] = True
    if args.max_loops_override is not None:
        gpt_model_config['max_loops'] = args.max_loops_override
    if 'effective_n_layer' not in gpt_model_config:
         gpt_model_config['effective_n_layer'] = None 
    
    # Ensure loop_groups is part of gpt_model_config if it exists in the checkpoint,
    # or set to None/empty if not, so plotting functions can check for it.
    # GPTConfig might create it as an attribute.
    if 'loop_groups' not in gpt_model_config:
        gpt_model_config['loop_groups'] = [] # Default to empty list if not present

    gptconf = GPTConfig(**gpt_model_config)
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix): state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.eval().to(device)
    print(f"Model: vocab {model.config.vocab_size}, block {model.config.block_size}, n_layer {model.config.n_layer}, max_loops {model.config.max_loops}")
    if model.config.loop_groups: print(f"Loop groups: {model.config.loop_groups}")

    # --- Plot Singular Values if requested ---
    if args.plot_singular_values:
        print("\nPlotting maximum singular values of model weights...")
        plot_max_singular_values(model, args.output_dir)
        if wandb_logging_enabled:
            wandb.log({"max_singular_values": wandb.Image(os.path.join(args.output_dir, "max_singular_values.png"))})

    print(f"Loading tokenizer from {args.meta_path}...")
    tokenizer_encode_fn = None
    # Function that takes one ID and returns its string representation
    tokenizer_decode_fn_for_single_id_to_str = None

    try:
        with open(args.meta_path, 'rb') as f:
            meta = pickle.load(f)

        # Try to use encode/decode methods from the loaded meta object first
        # meta is a dictionary, so check for keys and then if the values are callable
        if 'encode' in meta and callable(meta['encode']) and \
           'decode' in meta and callable(meta['decode']):
            print("Using encode/decode methods from meta.pkl (expected for BPE/SentencePiece).")
            tokenizer_encode_fn = meta['encode'] # Should take string, return list of IDs
            # meta.decode typically takes a list of IDs and returns a single string.
            # For prompt_tokens_str, we need string for each token.
            tokenizer_decode_fn_for_single_id_to_str = lambda token_id: meta['decode']([token_id])
        else:
            # Fallback to stoi/itos, assuming character-level if meta.encode/decode not found
            print("Warning: meta.pkl does not provide .encode/.decode methods. "
                  "Attempting to use 'stoi' and 'itos' from meta.pkl. "
                  "This is likely character-level tokenization if 'encode'/'decode' are missing.")
            if 'stoi' not in meta or 'itos' not in meta:
                print(f"Error: meta.pkl is missing 'stoi'/'itos' and also "
                      "lacks .encode/.decode methods. Cannot proceed with tokenization.")
                return
            
            stoi, itos = meta['stoi'], meta['itos']
            # This encode is character-by-character, as originally in the script
            tokenizer_encode_fn = lambda s: [stoi[c] for c in s if c in stoi]
            # This gets the string for a single ID
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
    # Use the selected encode function
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

    print("Getting loop representations...")
    with torch.no_grad():
        # Call generate, which might return more than 2 values now
        outputs = model.generate(
            input_tensor, 
            max_new_tokens=args.max_new_tokens_for_analysis, 
            return_first_step_loop_reps=True
        )

    # Unpack outputs carefully
    generated_ids = outputs[0] if isinstance(outputs, tuple) else outputs
    loop_representations_raw = None
    convergence_diagnostics = None
    jacobian_eigvals = None
    jacobian_eigval_trajectory = None
    global_diagnostics = None
    
    next_output_idx = 1
    if isinstance(outputs, tuple):
        # The first optional return is always loop_representations because return_first_step_loop_reps=True
        if len(outputs) > next_output_idx:
            loop_representations_raw = outputs[next_output_idx]
            next_output_idx += 1
        
        # Check for other optional returns based on config flags
        if model.config.track_convergence_diagnostics and len(outputs) > next_output_idx:
            convergence_diagnostics = outputs[next_output_idx]
            next_output_idx += 1

        if model.config.calculate_jacobian and len(outputs) > next_output_idx:
            jacobian_eigvals = outputs[next_output_idx]
            next_output_idx += 1
            
        if model.config.calculate_jacobian_trajectory and len(outputs) > next_output_idx:
            jacobian_eigval_trajectory = outputs[next_output_idx]
            next_output_idx += 1

        if model.config.track_global_diagnostics and len(outputs) > next_output_idx:
            global_diagnostics = outputs[next_output_idx]

    if not loop_representations_raw:
        print("Error: No loop representations returned."); return
    loop_representations_processed = [r.squeeze(0).cpu() for r in loop_representations_raw]
    print(f"Collected {len(loop_representations_processed)} sets of loop reps. Shape of first: {loop_representations_processed[0].shape if loop_representations_processed else 'N/A'}")
    prompt_seq_len = loop_representations_processed[0].shape[0] if loop_representations_processed else 0
    if prompt_seq_len == 0: print("Error: Zero sequence length from representations."); return

    if args.calculate_hausdorff_dimension:
        print("\nCalculating Hausdorff dimension for each token's trajectory (before PCA)...")
        if len(loop_representations_processed) > 1:
            hausdorff_dimensions = {}
            for token_idx in range(prompt_seq_len):
                trajectory_points = torch.stack([loop_representations_processed[i][token_idx] for i in range(len(loop_representations_processed))]).numpy()
                
                dim = box_counting_dimension(trajectory_points)
                
                token_str = prompt_tokens_str[token_idx] if token_idx < len(prompt_tokens_str) else f"UNK_{token_idx}"
                hausdorff_dimensions[f"token_{token_idx}_{token_str}"] = dim
                print(f"  Token '{token_str}' (pos {token_idx}): Estimated Hausdorff Dimension = {dim:.4f}")

            hausdorff_output_path = os.path.join(args.output_dir, "hausdorff_dimensions.txt")
            with open(hausdorff_output_path, 'w') as f:
                f.write("Estimated Hausdorff (Box-Counting) Dimensions:\n")
                for key, value in hausdorff_dimensions.items():
                    f.write(f"{key}: {value:.4f}\n")
            print(f"Hausdorff dimensions saved to {hausdorff_output_path}")
        else:
            print("Skipping Hausdorff dimension calculation: not enough loop representations (need > 1).")

    # --- Process and Plot Optional Diagnostics ---
    if convergence_diagnostics:
        print("\nPlotting convergence diagnostics...")
        diagnostics_output_dir = os.path.join(args.output_dir, "convergence_diagnostics_plots")
        plot_convergence_diagnostics(convergence_diagnostics, diagnostics_output_dir, model_config=gptconf)
        # Save raw data
        diagnostics_data_path = os.path.join(args.output_dir, "convergence_diagnostics.pkl")
        with open(diagnostics_data_path, 'wb') as f:
            pickle.dump(convergence_diagnostics, f)
        print(f"Convergence diagnostics data saved to {diagnostics_data_path}")
        if wandb_logging_enabled:
            for group_key in convergence_diagnostics.keys():
                plot_path = os.path.join(diagnostics_output_dir, f"convergence_diagnostics_{group_key}.png")
                if os.path.exists(plot_path):
                    wandb.log({f"convergence_diagnostics/{group_key}": wandb.Image(plot_path)})

    if jacobian_eigvals:
        print("\nPlotting Jacobian eigenvalues...")
        jacobian_output_dir = os.path.join(args.output_dir, "jacobian_eigenvalue_plots")
        plot_jacobian_eigenvalues(jacobian_eigvals, jacobian_output_dir, model_config=gptconf)
        # Save raw data
        jacobian_data_path = os.path.join(args.output_dir, "jacobian_eigenvalues.pkl")
        with open(jacobian_data_path, 'wb') as f:
            pickle.dump(jacobian_eigvals, f)
        print(f"Jacobian eigenvalues data saved to {jacobian_data_path}")
        if wandb_logging_enabled:
            for group_key in jacobian_eigvals.keys():
                plot_path = os.path.join(jacobian_output_dir, f"jacobian_eigvals_{group_key}.png")
                if os.path.exists(plot_path):
                    wandb.log({f"jacobian_eigenvalues/{group_key}": wandb.Image(plot_path)})

    if jacobian_eigval_trajectory:
        print("\nPlotting Jacobian eigenvalue trajectories...")
        jacobian_traj_output_dir = os.path.join(args.output_dir, "jacobian_eigenvalue_plots")
        plot_jacobian_eigenvalue_trajectory(jacobian_eigval_trajectory, jacobian_traj_output_dir)
        # Save raw data
        jacobian_traj_data_path = os.path.join(args.output_dir, "jacobian_eigval_trajectory.pkl")
        with open(jacobian_traj_data_path, 'wb') as f:
            pickle.dump(jacobian_eigval_trajectory, f)
        print(f"Jacobian eigenvalue trajectory data saved to {jacobian_traj_data_path}")
        if wandb_logging_enabled:
            for group_key in jacobian_eigval_trajectory.keys():
                plot_path = os.path.join(jacobian_traj_output_dir, f"jacobian_eigval_trajectory_{group_key}.png")
                if os.path.exists(plot_path):
                    wandb.log({f"jacobian_eigval_trajectory/{group_key}": wandb.Image(plot_path)})
        
    if global_diagnostics:
        print("\nPlotting global diagnostics...")
        global_diagnostics_output_dir = os.path.join(args.output_dir, "global_diagnostics_plots")
        plot_global_diagnostics(global_diagnostics, global_diagnostics_output_dir)
        # Save raw data
        global_diagnostics_data_path = os.path.join(args.output_dir, "global_diagnostics.pkl")
        with open(global_diagnostics_data_path, 'wb') as f:
            pickle.dump(global_diagnostics, f)
        print(f"Global diagnostics data saved to {global_diagnostics_data_path}")
        if wandb_logging_enabled:
            wandb.log({"global_diagnostics": wandb.Image(os.path.join(global_diagnostics_output_dir, "global_diagnostics.png"))})

    print(f"\nComputing PCA with up to {args.n_pca_components} components...")
    try:
        pca_model, transformed_reps_list = compute_pca_and_transform(loop_representations_processed, n_components=args.n_pca_components)
    except ValueError as e: print(f"Error during PCA: {e}"); return
    if not transformed_reps_list: print("PCA resulted in empty list."); return

    # --- Plotting ---
    # 2D Plots
    if args.n_pca_components >= 2:
        reps_for_plotting_2d = [arr[:, :2] for arr in transformed_reps_list]
        combined_plot_filename_2d = f"pca_trajectories_prompt_2D_pc{min(args.n_pca_components,2)}.png"
        combined_plot_filepath_2d = os.path.join(args.output_dir, combined_plot_filename_2d)
        print(f"Plotting combined 2D PCA trajectories to {combined_plot_filepath_2d}...")
        plot_pca_trajectories_2d(reps_for_plotting_2d, prompt_tokens_str, combined_plot_filepath_2d, model_config=gptconf)
        if wandb_logging_enabled:
            wandb.log({"pca_trajectories_2d_combined": wandb.Image(combined_plot_filepath_2d)})
    else:
        print("Skipping 2D plots as n_pca_components < 2.")

    # 3D Plots
    if args.n_pca_components >= 3:
        reps_for_plotting_3d = [arr[:, :3] for arr in transformed_reps_list]
        combined_plot_filename_3d = f"pca_trajectories_prompt_3D_pc{min(args.n_pca_components,3)}.png"
        combined_plot_filepath_3d = os.path.join(args.output_dir, combined_plot_filename_3d)
        print(f"Plotting combined 3D PCA trajectories to {combined_plot_filepath_3d}...")
        plot_pca_trajectories_3d(reps_for_plotting_3d, prompt_tokens_str, combined_plot_filepath_3d, model_config=gptconf)
        if wandb_logging_enabled:
            wandb.log({"pca_trajectories_3d_combined": wandb.Image(combined_plot_filepath_3d)})
    else:
        print("Skipping 3D plots as n_pca_components < 3.")

    individual_plots_dir = os.path.join(args.output_dir, "individual_token_plots")
    os.makedirs(individual_plots_dir, exist_ok=True)
    print(f"Plotting individual token PCA trajectories to {individual_plots_dir}...")

    for token_idx in range(prompt_seq_len):
        token_str = prompt_tokens_str[token_idx] if token_idx < len(prompt_tokens_str) else f"UNK_{token_idx}"
        sanitized_token_str = sanitize_filename_part(token_str if token_str != '?' else f"UNK_{token_idx}")
        num_available_loops = len(transformed_reps_list)

        # Individual 2D plots (full and zoomed)
        if args.n_pca_components >=2:
            filename_2d_full = f"token_{token_idx}_{sanitized_token_str}_pca_2D_full.png"
            filepath_2d_full = os.path.join(individual_plots_dir, filename_2d_full)
            plot_single_token_pca_trajectory(transformed_reps_list, token_idx, token_str, filepath_2d_full, 
                                             is_3d_plot=False, is_zoomed_view=False, model_config=gptconf)
            if wandb_logging_enabled:
                wandb.log({f"individual_plots/2d_full_token_{token_idx}": wandb.Image(filepath_2d_full)})

            if num_available_loops > args.num_last_steps_for_zoom:
                filename_2d_zoomed = f"token_{token_idx}_{sanitized_token_str}_pca_2D_zoomed_last{args.num_last_steps_for_zoom}.png"
                filepath_2d_zoomed = os.path.join(individual_plots_dir, filename_2d_zoomed)
                plot_single_token_pca_trajectory(transformed_reps_list, token_idx, token_str, filepath_2d_zoomed, 
                                                 is_3d_plot=False, is_zoomed_view=True, num_last_steps_to_zoom=args.num_last_steps_for_zoom, model_config=gptconf)
                if wandb_logging_enabled:
                    wandb.log({f"individual_plots/2d_zoomed_token_{token_idx}": wandb.Image(filepath_2d_zoomed)})
        
        # Individual 3D plots (full and zoomed)
        if args.n_pca_components >=3:
            filename_3d_full = f"token_{token_idx}_{sanitized_token_str}_pca_3D_full.png"
            filepath_3d_full = os.path.join(individual_plots_dir, filename_3d_full)
            plot_single_token_pca_trajectory(transformed_reps_list, token_idx, token_str, filepath_3d_full, 
                                             is_3d_plot=True, is_zoomed_view=False, model_config=gptconf)
            if wandb_logging_enabled:
                wandb.log({f"individual_plots/3d_full_token_{token_idx}": wandb.Image(filepath_3d_full)})

            if num_available_loops > args.num_last_steps_for_zoom:
                filename_3d_zoomed = f"token_{token_idx}_{sanitized_token_str}_pca_3D_zoomed_last{args.num_last_steps_for_zoom}.png"
                filepath_3d_zoomed = os.path.join(individual_plots_dir, filename_3d_zoomed)
                plot_single_token_pca_trajectory(transformed_reps_list, token_idx, token_str, filepath_3d_zoomed, 
                                                 is_3d_plot=True, is_zoomed_view=True, num_last_steps_for_zoom=args.num_last_steps_for_zoom, model_config=gptconf)
                if wandb_logging_enabled:
                    wandb.log({f"individual_plots/3d_zoomed_token_{token_idx}": wandb.Image(filepath_3d_zoomed)})

    print("Analysis complete.")

    if wandb_logging_enabled:
        wandb.finish()

if __name__ == '__main__':
    main() 