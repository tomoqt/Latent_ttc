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
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D # For 3D plotting
from sklearn.decomposition import PCA
import os
import pickle
import math
import re # For sanitizing filenames
import ast # For parsing string representations of python objects
import matplotlib.patches as mpatches # Added for custom legends
import json
from collections import defaultdict

import wandb
# Assuming model.py is in the same directory or accessible in PYTHONPATH
from model import GPTConfig, GPT

# Global plotting style
try:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 20,
        "legend.fontsize": 24,
    })
except Exception:
    # Fallback if LaTeX is not available in the environment
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "axes.titlesize": 24,
        "axes.labelsize": 24,
        "xtick.labelsize": 24,
        "ytick.labelsize": 20,
        "legend.fontsize": 24,
    })

sns.set_context("talk")

def box_counting_dimension(points):
    """
    Estimate the box-counting (Minkowski-Bouligand) dimension without PCA.
    Uses averaged counts over random grid shifts and fits only over a
    non-plateau scaling window.

    Args:
        points (np.ndarray): shape (n_points, n_dims)

    Returns:
        float: estimated dimension, or np.nan if no valid scaling window.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[0] < 2:
        return 0.0

    min_coords = points.min(axis=0)
    max_coords = points.max(axis=0)
    if np.all(min_coords == max_coords):
        return 0.0

    side_lengths = max_coords - min_coords
    points_normalized = (points - min_coords) / (side_lengths + 1e-12)

    n_points, n_dims = points_normalized.shape

    num_scales = 90
    scales = np.logspace(np.log10(1e-6), np.log10(1.0), num=num_scales, endpoint=False)
    num_shifts = 8
    rng = np.random.default_rng(0)

    counts = []
    for eps in scales:
        if eps <= 0:
            continue
        shift_counts = []
        for _ in range(num_shifts):
            offset = rng.uniform(0.0, eps, size=n_dims)
            bins = np.floor((points_normalized + offset) / eps)
            shift_counts.append(len(np.unique(bins, axis=0)))
        counts.append(float(np.mean(shift_counts)))

    counts = np.array(counts)

    # Keep scales away from both plateaus
    alpha = 0.9
    valid_mask = (counts > 1.0) & (counts < alpha * n_points)
    if np.count_nonzero(valid_mask) < 2:
        return np.nan

    valid_indices = np.where(valid_mask)[0]
    runs = []
    start = None
    prev = None
    for idx in valid_indices:
        if start is None:
            start = idx
            prev = idx
        elif idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = idx
            prev = idx
    if start is not None:
        runs.append((start, prev))

    if not runs:
        return np.nan

    run_lengths = [end - start + 1 for start, end in runs]
    best_run = runs[int(np.argmax(run_lengths))]
    start_i, end_i = best_run

    # Require a minimum number of scales for stability; otherwise fall back to all valid
    if (end_i - start_i + 1) < 4:
        start_i, end_i = valid_indices[0], valid_indices[-1]
        if (end_i - start_i + 1) < 2:
            return np.nan

    x = np.log(1.0 / scales[start_i:end_i + 1])
    y = np.log(counts[start_i:end_i + 1])
    coeffs = np.polyfit(x, y, 1)
    return float(coeffs[0])

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
    
    # Simplified plotting: no per-token lines, color by loop depth, marker shape by group
    depth_cmap = plt.cm.viridis
    group_markers = ['o','s','^','D','P','X','v','<','>']
    # Normalize loop index for color gradient
    total_loops_for_norm = total_model_loops if total_model_loops and total_model_loops > 0 else num_loops

    for loop_k in range(num_loops):
        group_idx = loop_k % num_loop_groups
        marker = group_markers[group_idx % len(group_markers)]
        depth_norm = (loop_k + 1) / max(1, total_loops_for_norm)
        color = depth_cmap(depth_norm)
        # scatter all tokens at this loop step
        pts = np.array([pca_transformed_reps_list[loop_k][token_idx, :2] for token_idx in range(seq_len)])
        ax.scatter(pts[:,0], pts[:,1], color=color, marker=marker, s=12, alpha=0.6, zorder=2 if loop_k>0 else 3,
                   label=f"Group {group_idx}" if loop_k < num_loop_groups else None)

    # Add colorbar to indicate loop depth
    sm = plt.cm.ScalarMappable(cmap=depth_cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Loop iteration depth')


    title_info = f'Combined 2D PCA Trajectories ({num_loops} Data Loops)'
    config_info = f'Model: {total_model_loops} Loops, Groups: {num_loop_groups}, Structure: {loop_group_str}'
    plt.title(f'{title_info}\n{config_info}', fontsize=10)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = {}
    for handle, label in zip(handles, labels):
        if label not in unique_labels : unique_labels[label] = handle
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='center left', bbox_to_anchor=(1, 0.5), fontsize='small')
    
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=0.5, alpha=0.5)
    plt.axvline(0, color='black', linewidth=0.5, alpha=0.5)

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
    # Simplified plotting: color by loop depth, marker shape by group, no per-token lines
    depth_cmap = plt.cm.viridis
    group_markers = ['o','s','^','D','P','X','v','<','>']
    total_loops_for_norm = total_model_loops if total_model_loops and total_model_loops > 0 else num_loops

    for loop_k in range(num_loops):
        group_idx = loop_k % num_loop_groups
        marker = group_markers[group_idx % len(group_markers)]
        depth_norm = (loop_k + 1) / max(1, total_loops_for_norm)
        color = depth_cmap(depth_norm)
        pts3 = np.array([pca_transformed_reps_list[loop_k][token_idx, :3] for token_idx in range(seq_len)])
        ax.scatter(pts3[:,0], pts3[:,1], pts3[:,2], color=color, marker=marker, s=12, alpha=0.6, zorder=2, depthshade=True,
                   label=f"Group {group_idx}" if loop_k < num_loop_groups else None)

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

    sm = plt.cm.ScalarMappable(cmap=depth_cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Loop iteration depth')
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

def plot_aggregated_convergence_diagnostics(aggregated_diagnostics_data, output_dir, model_name):
    """
    Plots the aggregated (mean/std) convergence diagnostics for a single model across all prompts.
    """
    if not aggregated_diagnostics_data:
        print("No aggregated convergence diagnostics data to plot for this model.")
        return

    os.makedirs(output_dir, exist_ok=True)

    saved_paths = {}
    for group_key, metrics in aggregated_diagnostics_data.items():
        if not metrics or 'delta_norm' not in metrics or 'mean' not in metrics['delta_norm'] or metrics['delta_norm']['mean'] is None:
            print(f"Skipping aggregated plot for {group_key} due to missing data.")
            continue

        fig, axs = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Aggregated Convergence Diagnostics for {model_name} - {group_key.replace("_", " ")}\n(Mean & Std Dev over prompts)', fontsize=16)

        metric_info = {
            'delta_norm': {'title': 'Change in Representation Norm (Δ-norm)', 'ylabel': r'$\|x_{k+1} - x_k\|_2$'},
            'delta_angle': {'title': 'Angle between Successive Changes (Δ-angle)', 'ylabel': r'$\cos\angle(\Delta_k, \Delta_{k-1})$'},
            'hidden_norm': {'title': 'Hidden Vector Norm', 'ylabel': r'$\|x_k\|$'},
            'logit_drift': {'title': 'Logit Drift (KL Divergence)', 'ylabel': r'KL$(p_{k-1}\|p_k)$'}
        }

        for ax, metric_key in zip(axs.flat, metric_info.keys()):
            info = metric_info[metric_key]
            ax.set_title(info['title'])

            if metric_key not in metrics or metrics[metric_key].get('mean') is None or len(metrics[metric_key]['mean']) == 0:
                ax.text(0.5, 0.5, 'No data available', ha='center', va='center')
                ax.grid(True)
                continue

            metric_agg = metrics[metric_key]
            mean_series = np.array(metric_agg.get('mean'))
            std_series = np.array(metric_agg.get('std'))

            ref_metric_len = len(metrics.get('delta_norm', {}).get('mean', []))
            start_iter = 1 if metric_key in ['delta_angle', 'logit_drift'] and ref_metric_len > len(mean_series) else 0
            iterations = np.arange(start_iter, start_iter + len(mean_series))

            line, = ax.plot(iterations, mean_series, marker='o', linestyle='-', markersize=4, label="Mean")
            if std_series is not None:
                ax.fill_between(iterations, mean_series - std_series, mean_series + std_series, color=line.get_color(), alpha=0.2, label="Std Dev")

            ax.set_xlabel('Loop Iteration')
            ax.set_ylabel(info['ylabel'])
            ax.grid(True, which="both" if metric_key == 'logit_drift' else "major")
            ax.legend()
            if metric_key == 'logit_drift':
                ax.set_yscale('log')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plot_filename = f"aggregated_convergence_diagnostics_{group_key}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)
        print(f"Aggregated convergence diagnostics plot saved to {plot_filepath}")
        saved_paths[group_key] = plot_filepath
    return saved_paths

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
        return max_singular_values

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
    return max_singular_values

def plot_comparison_hausdorff(all_models_results, output_dir):
    """
    Plots a comparison of Hausdorff dimensions across multiple models, with error bars for std dev.
    """
    if not all_models_results:
        print("No results to compare for Hausdorff dimensions.")
        return None

    models_with_data = {model_name: results for model_name, results in all_models_results.items() if 'hausdorff_dimensions' in results}

    if not models_with_data:
        print("No models have Hausdorff dimension data to compare.")
        return None

    all_token_positions = set()
    for model_name, results in models_with_data.items():
        if 'mean' in results['hausdorff_dimensions']:
            all_token_positions.update(results['hausdorff_dimensions']['mean'].keys())
    
    sorted_token_pos = sorted(list(all_token_positions), key=lambda x: int(x.split('_')[1]))
    num_tokens = len(sorted_token_pos)
    num_models = len(models_with_data)
    model_names = list(models_with_data.keys())

    plt.figure(figsize=(max(15, num_tokens * 1.5), 10))
    ax = plt.gca()

    bar_width = 0.8 / num_models
    index = np.arange(num_tokens)
    model_colors = plt.cm.get_cmap('tab10', num_models)

    # Color by checkpoint k with a gradient
    def _extract_ckpt_k(name):
        import re
        m = re.search(r'(\d+)(?!.*\d)', name)
        return int(m.group(1)) if m else None
    model_k = [(mn, _extract_ckpt_k(mn)) for mn in model_names]
    ks = [k for _, k in model_k if k is not None]
    k_min, k_max = (min(ks), max(ks)) if ks else (0, 1)
    cmap = plt.cm.viridis

    for i, (model_name, k) in enumerate(model_k):
        model_results = models_with_data[model_name]['hausdorff_dimensions']
        means = [model_results['mean'].get(pos, np.nan) for pos in sorted_token_pos]
        stds = [model_results['std'].get(pos, 0) for pos in sorted_token_pos]
        bar_positions = index + i * bar_width - (bar_width * (num_models -1) / 2)
        color = cmap(0.0 if k is None or k_max == k_min else (k - k_min) / (k_max - k_min))
        ax.bar(bar_positions, means, bar_width, yerr=stds, color=color, capsize=4)

    ax.set_xlabel('Token Position')
    ax.set_ylabel('Estimated Hausdorff Dimension (mean over prompts)')
    ax.set_title('Comparison of Trajectory Hausdorff Dimensions Across Models')
    ax.set_xticks(index)
    
    xtick_labels = [f"Pos {p.split('_')[1]}" for p in sorted_token_pos]
    ax.set_xticklabels(xtick_labels, rotation=45, ha="right")
    
    # Add colorbar for k
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Iterations (k)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = "comparison_hausdorff_dimensions.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Hausdorff dimension comparison plot saved to {plot_filepath}")
    return plot_filepath

def plot_comparison_singular_values(all_models_results, output_dir):
    """
    Plots a comparison of maximum singular values across multiple models.
    """
    if not all_models_results:
        print("No results to compare for singular values.")
        return None

    models_with_data = {model_name: results for model_name, results in all_models_results.items() if 'singular_values' in results}
    if not models_with_data:
        print("No models have singular value data to compare.")
        return None

    # Collect all parameter names
    all_param_names = set()
    for results in models_with_data.values():
        all_param_names.update(results['singular_values'].keys())
    
    sorted_param_names = sorted(list(all_param_names))
    num_params = len(sorted_param_names)
    num_models = len(models_with_data)
    model_names = list(models_with_data.keys())

    plt.figure(figsize=(max(15, num_params * 0.5), 12))
    ax = plt.gca()

    bar_width = 0.8 / num_models
    index = np.arange(num_params)
    model_colors = plt.cm.get_cmap('tab10', num_models)

    # Color by checkpoint k
    def _extract_ckpt_k(name):
        import re
        m = re.search(r'(\d+)(?!.*\d)', name)
        return int(m.group(1)) if m else None
    model_k = [(mn, _extract_ckpt_k(mn)) for mn in model_names]
    ks = [k for _, k in model_k if k is not None]
    k_min, k_max = (min(ks), max(ks)) if ks else (0, 1)
    cmap = plt.cm.viridis

    for i, (model_name, k) in enumerate(model_k):
        model_svs = models_with_data[model_name]['singular_values']
        values = [model_svs.get(param, np.nan) for param in sorted_param_names]
        bar_positions = index + i * bar_width - (bar_width * (num_models -1) / 2)
        color = cmap(0.0 if k is None or k_max == k_min else (k - k_min) / (k_max - k_min))
        ax.bar(bar_positions, values, bar_width, color=color)

    ax.set_xlabel('Model Parameter')
    ax.set_ylabel('Maximum Singular Value')
    ax.set_title('Comparison of Max Singular Values Across Models')
    ax.set_xticks(index)
    ax.set_xticklabels(sorted_param_names, rotation=90, fontsize='small')
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Iterations (k)')
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = "comparison_max_singular_values.png"
    plot_filepath = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_filepath, dpi=300)
    plt.close()
    print(f"Max singular value comparison plot saved to {plot_filepath}")
    return plot_filepath
def plot_hausdorff_dimensions_bar(hausdorff_means, hausdorff_stds, output_dir, model_name):
    """
    Create a bar plot of Hausdorff (box-count) dimension means with std error bars per token position.
    """
    if not hausdorff_means:
        return None

    os.makedirs(output_dir, exist_ok=True)
    positions = sorted(hausdorff_means.keys(), key=lambda x: int(x.split('_')[1]))
    means = [hausdorff_means[p] for p in positions]
    stds = [hausdorff_stds.get(p, 0.0) for p in positions] if hausdorff_stds else [0.0 for _ in positions]

    plt.figure(figsize=(max(12, len(positions) * 0.6), 6))
    x = np.arange(len(positions))
    plt.bar(x, means, yerr=stds, capsize=3)
    plt.xticks(x, [f"Pos {p.split('_')[1]}" for p in positions], rotation=45, ha='right')
    plt.ylabel('Estimated Hausdorff Dimension')
    plt.title(f'Hausdorff Dimensions per Token Position - {model_name}')
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    path = os.path.join(output_dir, 'hausdorff_dimensions.png')
    plt.savefig(path, dpi=300)
    plt.close()
    return path


def plot_comparison_convergence_diagnostics(all_models_results, output_dir):
    """
    Plots a comparison of convergence diagnostics across multiple models.
    """
    models_with_data = {name: res for name, res in all_models_results.items() if 'convergence_diagnostics' in res}
    if not models_with_data: return {}

    # Find all group keys and metrics across all models
    all_group_keys = set()
    for res in models_with_data.values():
        all_group_keys.update(res['convergence_diagnostics'].keys())

    if not all_group_keys: return {}

    # Define metrics to plot. Add more if needed.
    metric_keys = ['delta_norm', 'delta_angle', 'hidden_norm', 'logit_drift']
    metric_ylabels = {
        'delta_norm': r'Mean $\|x_{k+1} - x_k\|_2$', 'delta_angle': r'Mean $\cos\angle(\Delta_k, \Delta_{k-1})$',
        'hidden_norm': r'Mean $\|x_k\|$', 'logit_drift': r'Mean KL$(p_{k-1}\|p_k)$'
    }

    paths = {}
    for group_key in all_group_keys:
        for metric in metric_keys:
            plt.figure(figsize=(12, 8))
            ax = plt.gca()
            model_colors = plt.cm.get_cmap('tab10', len(models_with_data))

            for i, (model_name, results) in enumerate(models_with_data.items()):
                diag_data = results['convergence_diagnostics'].get(group_key)
                if diag_data and metric in diag_data and diag_data[metric]:
                    metric_agg = diag_data[metric]
                    mean_series = metric_agg.get('mean')
                    std_series = metric_agg.get('std')

                    if mean_series is not None and len(mean_series) > 0:
                        # Handle metrics that skip the first value (like delta_angle)
                        start_iter = 1 if metric in ['delta_angle', 'logit_drift'] and len(results['convergence_diagnostics'][group_key]['delta_norm']['mean']) > len(mean_series) else 0
                        iterations = np.arange(start_iter, start_iter + len(mean_series))
                        
                        line, = ax.plot(iterations, mean_series, marker='o', linestyle='-', markersize=4, label=model_name, color=model_colors(i))
                        if std_series is not None:
                            ax.fill_between(iterations, mean_series - std_series, mean_series + std_series, color=line.get_color(), alpha=0.2)

            ax.set_title(f'Comparison: {metric.replace("_", " ").title()} for {group_key.replace("_", " ")}')
            ax.set_xlabel('Loop Iteration')
            ax.set_ylabel(metric_ylabels.get(metric, 'Value'))
            ax.grid(True, which="both" if metric == 'logit_drift' else "major")
            if metric == 'logit_drift': ax.set_yscale('log')
            ax.legend(title="Models")
            plt.tight_layout()
            
            plot_filename = f"comparison_convergence_{group_key}_{metric}.png"
            plot_filepath = os.path.join(output_dir, plot_filename)
            plt.savefig(plot_filepath, dpi=300)
            plt.close()
            print(f"Saved convergence comparison plot to {plot_filepath}")
            paths[f"comparison_convergence_{group_key}_{metric}"] = plot_filepath
    return paths

def plot_comparison_jacobian_eigenvalues(all_models_results, output_dir):
    """
    Plots a comparison of Jacobian eigenvalues across multiple models.
    """
    models_with_data = {name: res for name, res in all_models_results.items() if 'jacobian_eigvals' in res}
    if not models_with_data: return {}

    all_group_keys = set().union(*(res['jacobian_eigvals'].keys() for res in models_with_data.values()))
    if not all_group_keys: return {}

    paths = {}
    for group_key in all_group_keys:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, aspect='equal')
        unit_circle = mpatches.Circle((0, 0), 1, color='black', fill=False, linestyle='--', alpha=0.5, label='Unit Circle')
        ax.add_patch(unit_circle)
        
        model_colors = plt.cm.get_cmap('tab10', len(models_with_data))
        max_abs_val = 1.0

        for i, (model_name, results) in enumerate(models_with_data.items()):
            token_eigvals_list = results['jacobian_eigvals'].get(group_key)
            if token_eigvals_list:
                all_eigvals = np.concatenate(token_eigvals_list)
                if all_eigvals.size > 0:
                    max_abs_val = max(max_abs_val, np.max(np.abs(all_eigvals)))
                    ax.scatter(all_eigvals.real, all_eigvals.imag, s=15, alpha=0.5, label=model_name, color=model_colors(i))
        
        lim = max_abs_val * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel('Real Part')
        ax.set_ylabel('Imaginary Part')
        ax.set_title(f'Comparison: Jacobian Eigenvalue Spectrum for {group_key.replace("_", " ")}')
        ax.grid(True)
        ax.axhline(0, color='grey', lw=0.5); ax.axvline(0, color='grey', lw=0.5)
        ax.legend(title="Models")
        plt.tight_layout()

        plot_filename = f"comparison_jacobian_eigvals_{group_key}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)
        print(f"Saved Jacobian eigenvalue comparison plot to {plot_filepath}")
        paths[f"comparison_jacobian_eigvals_{group_key}"] = plot_filepath
    return paths

def plot_comparison_jacobian_eigenvalue_trajectory(all_models_results, output_dir):
    """
    Plots a comparison of Jacobian eigenvalue trajectories across multiple models.
    """
    models_with_data = {name: res for name, res in all_models_results.items() if 'jacobian_eigval_trajectory' in res}
    if not models_with_data: return {}
    
    all_group_keys = set().union(*(res['jacobian_eigval_trajectory'].keys() for res in models_with_data.values()))
    if not all_group_keys: return {}

    paths = {}
    for group_key in all_group_keys:
        fig, ax = plt.subplots(figsize=(12, 8))
        model_colors = plt.cm.get_cmap('tab10', len(models_with_data))

        for i, (model_name, results) in enumerate(models_with_data.items()):
            traj_data = results['jacobian_eigval_trajectory'].get(group_key)
            if traj_data:
                mean_trajectory = traj_data.get('mean')
                std_trajectory = traj_data.get('std')
                if mean_trajectory is not None and len(mean_trajectory) > 0:
                    iterations = np.arange(len(mean_trajectory))
                    line, = ax.plot(iterations, mean_trajectory, marker='.', linestyle='-', label=model_name, color=model_colors(i))
                    if std_trajectory is not None:
                        ax.fill_between(iterations, mean_trajectory - std_trajectory, mean_trajectory + std_trajectory, color=line.get_color(), alpha=0.2)
        
        ax.axhline(1.0, color='r', linestyle='--', label='Stability Boundary (|λ|=1)')
        ax.set_xlabel('Loop Iteration')
        ax.set_ylabel('Mean Max Eigenvalue Magnitude |λ|')
        ax.set_title(f'Comparison: Mean Jacobian Max Eigenvalue Trajectory for {group_key.replace("_", " ")}')
        ax.grid(True)
        ax.legend(title="Models")
        plt.tight_layout()

        plot_filename = f"comparison_jacobian_eigval_trajectory_{group_key}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close(fig)
        print(f"Saved Jacobian eigenvalue trajectory comparison to {plot_filepath}")
        paths[f"comparison_jacobian_eigval_trajectory_{group_key}"] = plot_filepath
    return paths

def plot_comparison_global_diagnostics(all_models_results, output_dir):
    """
    Plots a comparison of global diagnostics across multiple models.
    """
    models_with_data = {name: res for name, res in all_models_results.items() if 'global_diagnostics' in res}
    if not models_with_data: return {}

    metric_keys = ['delta_norm', 'delta_angle', 'hidden_norm', 'logit_drift']
    metric_ylabels = {
        'delta_norm': r'Mean $\|x_{k+1} - x_k\|_2$', 'delta_angle': r'Mean $\cos\angle(\Delta_k, \Delta_{k-1})$',
        'hidden_norm': r'Mean $\|x_k\|$', 'logit_drift': r'Mean KL$(p_{k-1}\|p_k)$'
    }
    
    paths = {}
    for metric in metric_keys:
        plt.figure(figsize=(12, 8))
        ax = plt.gca()
        model_colors = plt.cm.get_cmap('tab10', len(models_with_data))

        for i, (model_name, results) in enumerate(models_with_data.items()):
            diag_data = results['global_diagnostics']
            if metric in diag_data and diag_data[metric]:
                metric_agg = diag_data[metric]
                mean_series = metric_agg.get('mean')
                std_series = metric_agg.get('std')
                if mean_series is not None and len(mean_series) > 0:
                    start_iter = 1 if metric in ['delta_angle', 'logit_drift'] and len(results['global_diagnostics']['delta_norm']['mean']) > len(mean_series) else 0
                    steps = np.arange(start_iter, start_iter + len(mean_series))
                    line, = ax.plot(steps, mean_series, marker='.', linestyle='-', markersize=4, label=model_name, color=model_colors(i))
                    if std_series is not None:
                        ax.fill_between(steps, mean_series - std_series, mean_series + std_series, color=line.get_color(), alpha=0.2)

        ax.set_title(f'Comparison: Global {metric.replace("_", " ").title()}')
        ax.set_xlabel('Global Step (Layer or Loop Iteration)')
        ax.set_ylabel(metric_ylabels.get(metric, 'Value'))
        ax.grid(True, which="both" if metric == 'logit_drift' else "major")
        if metric == 'logit_drift': ax.set_yscale('log')
        ax.legend(title="Models")
        plt.tight_layout()
        
        plot_filename = f"comparison_global_{metric}.png"
        plot_filepath = os.path.join(output_dir, plot_filename)
        plt.savefig(plot_filepath, dpi=300)
        plt.close()
        print(f"Saved global diagnostic comparison plot to {plot_filepath}")
        paths[f"comparison_global_{metric}"] = plot_filepath
    return paths

def plot_loop30_vs_checkpoint(all_models_results, output_dir, include_convergence=True, include_global=True):
    """
    For each metric, plot y = metric at loop index 30 (or last available) vs x = checkpoint (model name).
    Generates separate plots for convergence diagnostics (per group) and global diagnostics.
    """
    os.makedirs(output_dir, exist_ok=True)
    plots = {}

    # Helpers to extract checkpoint index (k) where ckpt_k ⇒ iterations = k (in thousands)
    def _extract_ckpt_k(name):
        import re
        m = re.search(r'(\d+)(?!.*\d)', name)
        return int(m.group(1)) if m else None

    # Helper to create a line plot with x = checkpoint k (thousands)
    def _line_plot_from_k(x_ks, y_values, title, ylabel, filename):
        # Sort by k
        pairs = sorted(zip(x_ks, y_values), key=lambda p: p[0])
        if not pairs:
            return None
        x_sorted = [p[0] for p in pairs]
        y_sorted = [p[1] for p in pairs]
        plt.figure(figsize=(max(10, len(x_sorted) * 0.9), 6))
        plt.plot(x_sorted, y_sorted, marker='o', linestyle='-')
        plt.xlabel('Iterations (k)')
        plt.ylabel(ylabel)
        plt.title(title)
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.tight_layout()
        path = os.path.join(output_dir, filename)
        plt.savefig(path, dpi=300)
        plt.close()
        return path

    # Convergence diagnostics (per group)
    if include_convergence and any('convergence_diagnostics' in res for res in all_models_results.values()):
        # Collect all group keys and metrics
        all_groups = set()
        for res in all_models_results.values():
            if 'convergence_diagnostics' in res:
                all_groups.update(res['convergence_diagnostics'].keys())
        metric_keys = ['delta_norm', 'delta_angle', 'hidden_norm', 'logit_drift']

        for group_key in sorted(all_groups):
            for metric in metric_keys:
                x_ks, y_vals = [], []
                for model_name, res in all_models_results.items():
                    if 'convergence_diagnostics' not in res: continue
                    gdata = res['convergence_diagnostics'].get(group_key)
                    if not gdata or metric not in gdata: continue
                    mean_series = gdata[metric].get('mean')
                    if mean_series is None or len(mean_series) == 0: continue
                    idx = min(30, len(mean_series) - 1)
                    val = mean_series[idx]
                    if val is None or (isinstance(val, float) and np.isnan(val)): continue
                    k = _extract_ckpt_k(model_name)
                    if k is None: continue
                    x_ks.append(k)
                    y_vals.append(float(val))
                if x_ks and y_vals:
                    fname = f"loop30_vs_ckpt_{group_key}_{metric}.png"
                    title = f"Loop-30 {metric} vs Checkpoint ({group_key})"
                    ylabel = metric.replace('_', ' ').title()
                    path = _line_plot_from_k(x_ks, y_vals, title, ylabel, fname)
                    plots[f"loop30_convergence_{group_key}_{metric}"] = path

    # Global diagnostics (single series)
    if include_global and any('global_diagnostics' in res for res in all_models_results.values()):
        metric_keys = ['delta_norm', 'delta_angle', 'hidden_norm', 'logit_drift']
        for metric in metric_keys:
            x_ks, y_vals = [], []
            for model_name, res in all_models_results.items():
                g = res.get('global_diagnostics')
                if not g or metric not in g: continue
                mean_series = g[metric].get('mean')
                if mean_series is None or len(mean_series) == 0: continue
                idx = min(30, len(mean_series) - 1)
                val = mean_series[idx]
                if val is None or (isinstance(val, float) and np.isnan(val)): continue
                k = _extract_ckpt_k(model_name)
                if k is None: continue
                x_ks.append(k)
                y_vals.append(float(val))
            if x_ks and y_vals:
                fname = f"loop30_vs_ckpt_global_{metric}.png"
                title = f"Loop-30 {metric} vs Checkpoint (Global)"
                ylabel = metric.replace('_', ' ').title()
                path = _line_plot_from_k(x_ks, y_vals, title, ylabel, fname)
                plots[f"loop30_global_{metric}"] = path

    return plots

def _load_tables_from_dir(tables_dir):
    """
    Load metrics tables from a directory previously produced by this script.
    Expected files:
      - convergence_{group_key}.csv   (columns: loop_iter, <metrics...>)
      - global_diagnostics.csv        (columns: loop_iter, <metrics...>)
      - optionally: hausdorff_dimensions.csv (columns: pos, mean, std)
    Returns a results dict in the same schema used by analyze_single_model.
    """
    results = {}
    if not os.path.isdir(tables_dir):
        return results

    # Convergence diagnostics (per group)
    conv = {}
    try:
        for fname in os.listdir(tables_dir):
            if fname.startswith('convergence_') and fname.endswith('.csv'):
                group_key = fname[len('convergence_'):-len('.csv')]
                path = os.path.join(tables_dir, fname)
                with open(path, 'r') as f:
                    header = f.readline().strip().split(',')
                    rows = f.read().strip().splitlines()
                # header[0] is loop_iter; others are metric keys
                metric_keys = header[1:]
                series = {k: [] for k in metric_keys}
                for line in rows:
                    parts = line.split(',')
                    for i, k in enumerate(metric_keys, start=1):
                        val = parts[i].strip()
                        if val == '':
                            series[k].append(None)
                        else:
                            try:
                                series[k].append(float(val))
                            except ValueError:
                                series[k].append(None)
                conv[group_key] = {mk: {'mean': series[mk]} for mk in metric_keys}
    except Exception as e:
        print(f"Warning: failed loading convergence tables from {tables_dir}: {e}")

    if conv:
        results['convergence_diagnostics'] = conv

    # Global diagnostics
    try:
        gpath = os.path.join(tables_dir, 'global_diagnostics.csv')
        if os.path.isfile(gpath):
            with open(gpath, 'r') as f:
                header = f.readline().strip().split(',')
                rows = f.read().strip().splitlines()
            metric_keys = header[1:]
            series = {k: [] for k in metric_keys}
            for line in rows:
                parts = line.split(',')
                for i, k in enumerate(metric_keys, start=1):
                    val = parts[i].strip()
                    if val == '':
                        series[k].append(None)
                    else:
                        try:
                            series[k].append(float(val))
                        except ValueError:
                            series[k].append(None)
            results['global_diagnostics'] = {mk: {'mean': series[mk]} for mk in metric_keys}
    except Exception as e:
        print(f"Warning: failed loading global diagnostics from {tables_dir}: {e}")

    # Optional Hausdorff (if present)
    try:
        hpath = os.path.join(tables_dir, 'hausdorff_dimensions.csv')
        if os.path.isfile(hpath):
            import csv
            means, stds = {}, {}
            with open(hpath, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    pos = row.get('pos') or row.get('position') or row.get('token')
                    mean_s = row.get('mean')
                    std_s = row.get('std')
                    if pos and mean_s:
                        means[pos] = float(mean_s)
                        stds[pos] = float(std_s) if std_s else 0.0
            if means:
                results['hausdorff_dimensions'] = {'mean': means, 'std': stds}
    except Exception as e:
        print(f"Warning: failed loading Hausdorff dimensions from {tables_dir}: {e}")

    return results

def analyze_single_model(checkpoint_path, output_dir, model_name, args, config_overrides, prompts, tokenizer_encode_fn, tokenizer_decode_fn_for_single_id_to_str, wandb_logging_enabled):
    """
    Performs a full analysis for a single model checkpoint over a list of prompts, aggregating the results.
    """
    device = torch.device(args.device)
    results_for_comparison = {}

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    if 'model_args' not in checkpoint:
        print(f"Error: 'model_args' not found in checkpoint {checkpoint_path}. Skipping.")
        return None

    gpt_model_config = checkpoint['model_args']

    # Apply overrides for this model
    if config_overrides:
        print("Applying overrides for this model...")
        for key, value in config_overrides.items():
            if not key.startswith('__'):
                print(f"  Overriding: {key} = {value}")
                gpt_model_config[key] = value

    # Set analysis-specific config flags
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
        print(f"  Overriding from CLI: max_loops = {gpt_model_config['max_loops']}")

    # Finalize model config
    if 'effective_n_layer' not in gpt_model_config:
        gpt_model_config['effective_n_layer'] = None
    if 'loop_groups' not in gpt_model_config:
        gpt_model_config['loop_groups'] = []

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

    # --- Singular values plot (prompt-independent) ---
    if args.plot_singular_values:
        print("\nPlotting maximum singular values of model weights...")
        max_sv_data = plot_max_singular_values(model, output_dir)
        results_for_comparison['singular_values'] = max_sv_data
        if wandb_logging_enabled:
            wandb.log({f"{model_name}/max_singular_values": wandb.Image(os.path.join(output_dir, "max_singular_values.png"))})

    # Data structures for aggregating results across prompts
    all_hausdorff_dims = defaultdict(list)
    all_conv_diags = defaultdict(lambda: defaultdict(list))
    all_jacobian_eigvals = defaultdict(list)
    all_jacobian_trajs = defaultdict(lambda: defaultdict(list))
    all_global_diags = defaultdict(list)

    # --- Loop over prompts to collect data ---
    print(f"\nAnalyzing model over {len(prompts)} prompts...")
    for i, prompt_text in enumerate(prompts):
        print(f"  Processing prompt {i+1}/{len(prompts)}: \"{prompt_text[:50]}...\"")

        input_ids = tokenizer_encode_fn(prompt_text)
        if not input_ids:
            print(f"Warning: Could not tokenize prompt. Skipping."); continue
        if len(input_ids) > model.config.block_size:
            input_ids = input_ids[:model.config.block_size]
        prompt_tokens_str = [tokenizer_decode_fn_for_single_id_to_str(id_) for id_ in input_ids]
        input_tensor = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

        with torch.no_grad():
            outputs = model.generate(input_tensor, max_new_tokens=args.max_new_tokens_for_analysis, return_first_step_loop_reps=True)
        
        # Unpack model outputs
        loop_representations_raw, convergence_diagnostics, jacobian_eigvals, jacobian_eigval_trajectory, global_diagnostics = None, None, None, None, None
        next_output_idx = 1
        if isinstance(outputs, tuple):
            if len(outputs) > next_output_idx: loop_representations_raw = outputs[next_output_idx]; next_output_idx += 1
            if model.config.track_convergence_diagnostics and len(outputs) > next_output_idx: convergence_diagnostics = outputs[next_output_idx]; next_output_idx += 1
            if model.config.calculate_jacobian and len(outputs) > next_output_idx: jacobian_eigvals = outputs[next_output_idx]; next_output_idx += 1
            if model.config.calculate_jacobian_trajectory and len(outputs) > next_output_idx: jacobian_eigval_trajectory = outputs[next_output_idx]; next_output_idx += 1
            if model.config.track_global_diagnostics and len(outputs) > next_output_idx: global_diagnostics = outputs[next_output_idx]

        if not loop_representations_raw: continue
        loop_reps = [r.squeeze(0).cpu() for r in loop_representations_raw]
        prompt_seq_len = loop_reps[0].shape[0] if loop_reps else 0
        if prompt_seq_len == 0: continue

        # --- Aggregate data for this prompt ---
        if args.calculate_hausdorff_dimension and len(loop_reps) > 1:
            use_pca_for_hd = bool(getattr(args, 'hausdorff_after_pca', False))
            transformed_reps_list_for_hd = None
            if use_pca_for_hd:
                # Ensure at least 2 components when using PCA for Hausdorff
                n_comp_hd = max(2, int(getattr(args, 'n_pca_components', 2)))
                try:
                    _, transformed_reps_list_for_hd = compute_pca_and_transform(loop_reps, n_components=n_comp_hd)
                except Exception as e:
                    print(f"Warning: PCA for Hausdorff failed ({e}). Falling back to original space.")
                    use_pca_for_hd = False

            for token_idx in range(prompt_seq_len):
                if use_pca_for_hd and transformed_reps_list_for_hd is not None:
                    # Build trajectory in PCA space (use all available components from the transform)
                    num_comp_avail = transformed_reps_list_for_hd[0].shape[1]
                    traj_points = np.stack([
                        transformed_reps_list_for_hd[i][token_idx, :num_comp_avail]
                        for i in range(len(transformed_reps_list_for_hd))
                    ], axis=0)
                else:
                    traj_points = torch.stack([loop_reps[i][token_idx] for i in range(len(loop_reps))]).numpy()

                dim = box_counting_dimension(traj_points)
                all_hausdorff_dims[f"pos_{token_idx}"].append(dim)
                try:
                    token_label = prompt_tokens_str[token_idx]
                except Exception:
                    token_label = f"pos_{token_idx}"
                suffix = " (PCA space)" if use_pca_for_hd and transformed_reps_list_for_hd is not None else ""
                print(f"Hausdorff (box-count) dim for token {token_idx} '{token_label}': {dim}{suffix}")

        if convergence_diagnostics:
            for group_key, metrics in convergence_diagnostics.items():
                for metric_key, values in metrics.items():
                    all_conv_diags[group_key][metric_key].append(values)
        
        if jacobian_eigvals:
            for group_key, eig_list in jacobian_eigvals.items():
                all_jacobian_eigvals[group_key].extend(eig_list)
        
        if jacobian_eigval_trajectory:
            for group_key, traj_list in jacobian_eigval_trajectory.items():
                all_jacobian_trajs[group_key][i].append(traj_list)

        if global_diagnostics:
            for metric_key, values in global_diagnostics.items():
                all_global_diags[metric_key].append(values)


        # --- Generate detailed plots only for the first prompt (skip when only Hausdorff is requested) ---
        if i == 0 and not getattr(args, 'only_hausdorff', False) and not getattr(args, 'only_loop30_metrics', False):
            print("  (Generating detailed plots for the first prompt only)")
            if loop_reps:
                try:
                    pca_model, transformed_reps_list = compute_pca_and_transform(loop_reps, n_components=args.n_pca_components)
                    if not transformed_reps_list: raise ValueError("PCA resulted in empty list.")
                    
                    pca2d_path, pca3d_path = None, None
                    pca2d_zoom_path, pca3d_zoom_path = None, None
                    if args.n_pca_components >= 2:
                        pca2d_path = os.path.join(output_dir, "pca_trajectories_prompt_2D.png")
                        plot_pca_trajectories_2d([arr[:, :2] for arr in transformed_reps_list], prompt_tokens_str, pca2d_path, model_config=gptconf)
                    if args.n_pca_components >= 3:
                        pca3d_path = os.path.join(output_dir, "pca_trajectories_prompt_3D.png")
                        plot_pca_trajectories_3d([arr[:, :3] for arr in transformed_reps_list], prompt_tokens_str, pca3d_path, model_config=gptconf)

                    # Zoomed combined trajectories (last K loops)
                    last_k = max(1, min(int(getattr(args, 'num_last_steps_for_zoom', 15)), len(transformed_reps_list)))
                    if args.n_pca_components >= 2 and last_k < len(transformed_reps_list):
                        pca2d_zoom_path = os.path.join(output_dir, f"pca_trajectories_prompt_2D_zoomed_last{last_k}.png")
                        plot_pca_trajectories_2d([arr[:, :2] for arr in transformed_reps_list[-last_k:]], prompt_tokens_str, pca2d_zoom_path, model_config=gptconf)
                    if args.n_pca_components >= 3 and last_k < len(transformed_reps_list):
                        pca3d_zoom_path = os.path.join(output_dir, f"pca_trajectories_prompt_3D_zoomed_last{last_k}.png")
                        plot_pca_trajectories_3d([arr[:, :3] for arr in transformed_reps_list[-last_k:]], prompt_tokens_str, pca3d_zoom_path, model_config=gptconf)
                    
                    individual_plots_dir = os.path.join(output_dir, "individual_token_plots")
                    os.makedirs(individual_plots_dir, exist_ok=True)
                    for token_idx in range(prompt_seq_len):
                        sanitized_token_str = sanitize_filename_part(prompt_tokens_str[token_idx])
                        if args.n_pca_components >=2:
                            plot_single_token_pca_trajectory(transformed_reps_list, token_idx, prompt_tokens_str[token_idx], os.path.join(individual_plots_dir, f"token_{token_idx}_{sanitized_token_str}_pca_2D_full.png"), is_3d_plot=False, model_config=gptconf)
                        if args.n_pca_components >=3:
                            plot_single_token_pca_trajectory(transformed_reps_list, token_idx, prompt_tokens_str[token_idx], os.path.join(individual_plots_dir, f"token_{token_idx}_{sanitized_token_str}_pca_3D_full.png"), is_3d_plot=True, model_config=gptconf)

                    # Zoomed individual token plots for a subset (last K loops)
                    max_token_logs = min(5, prompt_seq_len)
                    for token_idx in range(max_token_logs):
                        sanitized_token_str = sanitize_filename_part(prompt_tokens_str[token_idx])
                        if args.n_pca_components >=2 and last_k < len(transformed_reps_list):
                            plot_single_token_pca_trajectory(transformed_reps_list, token_idx, prompt_tokens_str[token_idx], os.path.join(individual_plots_dir, f"token_{token_idx}_{sanitized_token_str}_pca_2D_zoomed_last{last_k}.png"), is_3d_plot=False, is_zoomed_view=True, num_last_steps_to_zoom=last_k, model_config=gptconf)
                        if args.n_pca_components >=3 and last_k < len(transformed_reps_list):
                            plot_single_token_pca_trajectory(transformed_reps_list, token_idx, prompt_tokens_str[token_idx], os.path.join(individual_plots_dir, f"token_{token_idx}_{sanitized_token_str}_pca_3D_zoomed_last{last_k}.png"), is_3d_plot=True, is_zoomed_view=True, num_last_steps_to_zoom=last_k, model_config=gptconf)

                    # Log trajectory plots to WandB (combined, and a small sample of individual tokens)
                    if wandb_logging_enabled:
                        wandb_payload = {}
                        if pca2d_path and os.path.exists(pca2d_path):
                            wandb_payload[f"{model_name}/pca_trajectories/combined_2D_first_prompt"] = wandb.Image(pca2d_path)
                        if pca3d_path and os.path.exists(pca3d_path):
                            wandb_payload[f"{model_name}/pca_trajectories/combined_3D_first_prompt"] = wandb.Image(pca3d_path)
                        if pca2d_zoom_path and os.path.exists(pca2d_zoom_path):
                            wandb_payload[f"{model_name}/pca_trajectories/combined_2D_zoom_first_prompt"] = wandb.Image(pca2d_zoom_path)
                        if pca3d_zoom_path and os.path.exists(pca3d_zoom_path):
                            wandb_payload[f"{model_name}/pca_trajectories/combined_3D_zoom_first_prompt"] = wandb.Image(pca3d_zoom_path)
                        # Log up to first 5 token plots (2D if available else 3D)
                        max_token_logs = min(5, prompt_seq_len)
                        for token_idx in range(max_token_logs):
                            sanitized_token_str = sanitize_filename_part(prompt_tokens_str[token_idx])
                            tok2d = os.path.join(individual_plots_dir, f"token_{token_idx}_{sanitized_token_str}_pca_2D_full.png")
                            tok3d = os.path.join(individual_plots_dir, f"token_{token_idx}_{sanitized_token_str}_pca_3D_full.png")
                            tok2d_zoom = os.path.join(individual_plots_dir, f"token_{token_idx}_{sanitized_token_str}_pca_2D_zoomed_last{last_k}.png")
                            tok3d_zoom = os.path.join(individual_plots_dir, f"token_{token_idx}_{sanitized_token_str}_pca_3D_zoomed_last{last_k}.png")
                            if os.path.exists(tok2d):
                                wandb_payload[f"{model_name}/pca_trajectories/token_{token_idx}_2D_first_prompt"] = wandb.Image(tok2d)
                            elif os.path.exists(tok3d):
                                wandb_payload[f"{model_name}/pca_trajectories/token_{token_idx}_3D_first_prompt"] = wandb.Image(tok3d)
                            if os.path.exists(tok2d_zoom):
                                wandb_payload[f"{model_name}/pca_trajectories/token_{token_idx}_2D_zoom_first_prompt"] = wandb.Image(tok2d_zoom)
                            elif os.path.exists(tok3d_zoom):
                                wandb_payload[f"{model_name}/pca_trajectories/token_{token_idx}_3D_zoom_first_prompt"] = wandb.Image(tok3d_zoom)
                        if wandb_payload:
                            wandb.log(wandb_payload)
                except (ValueError, IndexError) as e:
                    print(f"  Warning: Could not generate PCA plots for first prompt: {e}")
            
            if convergence_diagnostics: plot_convergence_diagnostics(convergence_diagnostics, os.path.join(output_dir, "convergence_diagnostics_plots_first_prompt"), model_config=gptconf)
            if jacobian_eigvals: plot_jacobian_eigenvalues(jacobian_eigvals, os.path.join(output_dir, "jacobian_eigenvalue_plots_first_prompt"), model_config=gptconf)
            if jacobian_eigval_trajectory: plot_jacobian_eigenvalue_trajectory(jacobian_eigval_trajectory, os.path.join(output_dir, "jacobian_eigenvalue_plots_first_prompt"))
            if global_diagnostics: plot_global_diagnostics(global_diagnostics, os.path.join(output_dir, "global_diagnostics_plots_first_prompt"))


    # --- Aggregate results across all prompts ---
    print("\nAggregating results across all prompts...")
    if args.calculate_hausdorff_dimension and all_hausdorff_dims:
        hd_means = {pos: np.mean(dims) for pos, dims in all_hausdorff_dims.items()}
        hd_stds = {pos: np.std(dims) for pos, dims in all_hausdorff_dims.items()}
        results_for_comparison['hausdorff_dimensions'] = {
            'mean': hd_means,
            'std': hd_stds
        }
        # Per-checkpoint WandB logging (numerical and plot)
        plot_path = plot_hausdorff_dimensions_bar(hd_means, hd_stds, output_dir, model_name)
        if wandb_logging_enabled:
            log_dict = {}
            if plot_path and os.path.exists(plot_path):
                log_dict[f"{model_name}/hausdorff_dimensions_plot"] = wandb.Image(plot_path)
            # Log scalar metrics per token position
            for pos_key, mean_val in hd_means.items():
                log_dict[f"{model_name}/hausdorff/{pos_key}_mean"] = float(mean_val)
            for pos_key, std_val in hd_stds.items():
                log_dict[f"{model_name}/hausdorff/{pos_key}_std"] = float(std_val)
            if log_dict:
                wandb.log(log_dict)
    
    def aggregate_diagnostic_data(all_data_by_key):
        agg_results = {}
        for key, list_of_series in all_data_by_key.items():
            if not list_of_series: continue

            print(f"DEBUG: Aggregating for metric: {key}. Number of prompts/series: {len(list_of_series)}")

            # Replace None with np.nan before padding and convert to float
            series_with_nan = []
            for s in list_of_series:
                 if hasattr(s, '__iter__'):
                    series_with_nan.append([float(item) if item is not None else np.nan for item in s])
                 elif s is not None:
                    series_with_nan.append([float(s)])
                 else:
                    series_with_nan.append([np.nan])

            # Pad series to the same length (max length)
            max_len = max(len(s) for s in series_with_nan if hasattr(s, '__len__'))
            padded_series = [np.pad(s, (0, max_len - len(s)), 'constant', constant_values=np.nan) for s in series_with_nan]
            
            if not padded_series: continue
            stacked_series = np.array(padded_series)
            print(f"DEBUG:   - Shape of stacked_series for {key}: {stacked_series.shape}")
            
            # Check if there is any data to aggregate to avoid warnings on all-NaN slices
            if np.all(np.isnan(stacked_series)):
                continue
            
            with np.errstate(invalid='ignore'):
                mean_vals = np.nanmean(stacked_series, axis=0)
            
            # Calculate std dev only where there's more than one data point to avoid warnings
            n_valid = np.count_nonzero(~np.isnan(stacked_series), axis=0)
            std_vals = np.full_like(mean_vals, 0.0) # Default std to 0
            
            # Identify columns with enough data for a meaningful std dev calculation
            sufficient_data_mask = n_valid > 1
            if np.any(sufficient_data_mask):
                # Calculate std only for those columns
                std_vals[sufficient_data_mask] = np.nanstd(stacked_series[:, sufficient_data_mask], axis=0)

            print(f"DEBUG:   - Shapes after aggregation for {key}: mean={mean_vals.shape}, std={std_vals.shape}, n_valid={n_valid.shape}")

            agg_results[key] = {
                'mean': mean_vals,
                'std': std_vals
            }
        return agg_results

    if args.track_convergence_diagnostics and all_conv_diags:
        agg_data = {}
        for group_key, metrics in all_conv_diags.items():
            agg_data[group_key] = aggregate_diagnostic_data(metrics)
        results_for_comparison['convergence_diagnostics'] = agg_data
    
    if args.track_global_diagnostics and all_global_diags:
        results_for_comparison['global_diagnostics'] = aggregate_diagnostic_data(all_global_diags)

    if args.calculate_jacobian_trajectory and all_jacobian_trajs:
        agg_data = {}
        for group_key, prompts_data in all_jacobian_trajs.items():
            # prompts_data is dict {prompt_idx: [[token_trajs]]}
            mean_trajs_across_prompts = []
            for p_idx, traj_lists in prompts_data.items():
                # Average over tokens for this prompt
                token_trajectories = traj_lists[0] # [[traj_tok1], [traj_tok2], ...]
                if token_trajectories:
                    max_len = max(len(t) for t in token_trajectories)
                    padded = [np.pad(t, (0, max_len - len(t)), 'constant', constant_values=np.nan) for t in token_trajectories]
                    mean_trajs_across_prompts.append(np.nanmean(np.array(padded), axis=0))
            
            if mean_trajs_across_prompts:
                # Average over prompts
                max_len_means = max(len(t) for t in mean_trajs_across_prompts)
                padded_means = [np.pad(t, (0, max_len_means - len(t)), 'constant', constant_values=np.nan) for t in mean_trajs_across_prompts]
                stacked = np.array(padded_means)
                agg_data[group_key] = {
                    'mean': np.nanmean(stacked, axis=0),
                    'std': np.nanstd(stacked, axis=0)
                }
        results_for_comparison['jacobian_eigval_trajectory'] = agg_data

    # For eigenvalues, just collect all of them for the comparison plot
    if args.calculate_jacobian and all_jacobian_eigvals:
        results_for_comparison['jacobian_eigvals'] = dict(all_jacobian_eigvals)

    # --- Plot aggregated results for this single model ---
    aggregated_output_dir = os.path.join(output_dir, "aggregated_plots")
    os.makedirs(aggregated_output_dir, exist_ok=True)
    
    if args.track_convergence_diagnostics and 'convergence_diagnostics' in results_for_comparison:
        print("\nPlotting aggregated convergence diagnostics for this model...")
        paths = plot_aggregated_convergence_diagnostics(
            results_for_comparison['convergence_diagnostics'],
            aggregated_output_dir,
            model_name
        )
        if wandb_logging_enabled and paths:
            log_dict = {}
            for group_key, path in paths.items():
                if path and os.path.exists(str(path)):
                    log_dict[f"{model_name}/aggregated_convergence/{group_key}"] = wandb.Image(path)
            if log_dict:
                wandb.log(log_dict)

    # Log a per-checkpoint summary metric table for loop-30 values across metrics
    try:
        if wandb_logging_enabled:
            # CSV/Tables for metrics (loop series and loop-30 summary)
            csv_output_dir = os.path.join(output_dir, "tables")
            os.makedirs(csv_output_dir, exist_ok=True)
            log_dict = {}

            # Convergence diagnostics (per group)
            if args.track_convergence_diagnostics and 'convergence_diagnostics' in results_for_comparison:
                for group_key, metrics in results_for_comparison['convergence_diagnostics'].items():
                    # loop-30 summary
                    for metric_key, series in metrics.items():
                        if 'mean' in series and isinstance(series['mean'], (list, np.ndarray)) and len(series['mean']) > 0:
                            idx = min(30, len(series['mean']) - 1)
                            log_dict[f"{model_name}/loop30/{group_key}/{metric_key}"] = float(series['mean'][idx])
                    # save full series as CSV
                    import csv
                    csv_path = os.path.join(csv_output_dir, f"convergence_{group_key}.csv")
                    keys = sorted(metrics.keys())
                    # compute max length across metrics
                    max_len = 0
                    for k in keys:
                        s = metrics[k].get('mean', [])
                        max_len = max(max_len, len(s) if s is not None else 0)
                    rows_for_table = []
                    with open(csv_path, 'w', newline='') as f:
                        writer = csv.writer(f)
                        header = ['loop_iter'] + keys
                        writer.writerow(header)
                        for i in range(max_len):
                            row = [i] + [metrics[k]['mean'][i] if ('mean' in metrics[k] and i < len(metrics[k]['mean'])) else '' for k in keys]
                            writer.writerow(row)
                            rows_for_table.append(row)
                    try:
                        log_dict[f"{model_name}/tables/convergence_{group_key}"] = wandb.Table(data=rows_for_table, columns=header)
                    except Exception as te:
                        print(f"Warning: failed creating WandB Table for convergence {group_key}: {te}")

            # Global diagnostics
            if args.track_global_diagnostics and 'global_diagnostics' in results_for_comparison:
                import csv
                g = results_for_comparison['global_diagnostics']
                csv_path = os.path.join(csv_output_dir, f"global_diagnostics.csv")
                keys = sorted(g.keys())
                max_len = 0
                for k in keys:
                    s = g[k].get('mean', [])
                    max_len = max(max_len, len(s) if s is not None else 0)
                rows_for_table = []
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    header = ['loop_iter'] + keys
                    writer.writerow(header)
                    for i in range(max_len):
                        row = [i] + [g[k]['mean'][i] if ('mean' in g[k] and i < len(g[k]['mean'])) else '' for k in keys]
                        writer.writerow(row)
                        rows_for_table.append(row)
                try:
                    log_dict[f"{model_name}/tables/global_diagnostics"] = wandb.Table(data=rows_for_table, columns=header)
                except Exception as te:
                    print(f"Warning: failed creating WandB Table for global diagnostics: {te}")

            if log_dict:
                wandb.log(log_dict)
    except Exception as e:
        print(f"Warning: failed to log loop-30 metrics/tables: {e}")

    return results_for_comparison


def main():
    parser = argparse.ArgumentParser(description="Analyze and visualize loop representations from a GPT model using PCA.")
    parser.add_argument('--checkpoint_paths', type=str, nargs='+', required=False, help='One or more full paths to model checkpoint (.pt files)')
    parser.add_argument('--tables_dirs', type=str, nargs='+', default=None, help='One or more directories with precomputed tables (CSV) to plot from, bypassing inference.')
    parser.add_argument('--model_configs', type=str, nargs='+', default=None, help='A list of JSON strings or file paths to Python config files, one for each checkpoint.')
    parser.add_argument('--prompts_file', type=str, default=None, help='Path to a text file containing prompts, one per line. If not provided, a default prompt is used.')
    parser.add_argument('--prompt', type=str, default="Hello world, this is a test.", help='Input prompt string (used if prompts_file is not provided)')
    parser.add_argument('--output_dir', type=str, default='representation_analysis_output', help='Directory to save plots')
    parser.add_argument('--meta_path', type=str, default='data/fineweb/meta.pkl', help='Path to meta.pkl for tokenizer')
    parser.add_argument('--max_loops_override', type=int, default=None, help='Global override for model config max_loops for all models.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use')
    parser.add_argument('--n_pca_components', type=int, default=2, help='Number of PCA components (2 or 3 for plotting)')
    parser.add_argument('--max_new_tokens_for_analysis', type=int, default=1, help='Number of new tokens for representation collection trigger')
    parser.add_argument('--num_last_steps_for_zoom', type=int, default=15, help='Number of last loops for zoomed plots')
    parser.add_argument('--calculate_hausdorff_dimension', action='store_true', help='If set, calculate the Hausdorff dimension of trajectories.')
    parser.add_argument('--hausdorff_after_pca', action='store_true', help='If set, compute Hausdorff (box-count) on PCA-transformed trajectories.')
    parser.add_argument('--track_convergence_diagnostics', action='store_true', help='If set, track and plot convergence diagnostics.')
    parser.add_argument('--calculate_jacobian', action='store_true', help='If set, calculate and plot Jacobian eigenvalues.')
    parser.add_argument('--calculate_jacobian_trajectory', action='store_true', help='If set, plot the trajectory of the max Jacobian eigenvalue.')
    parser.add_argument('--track_global_diagnostics', action='store_true', help='If set, plot diagnostics across the entire model forward pass.')
    parser.add_argument('--plot_singular_values', action='store_true', help='If set, plot the max singular values of the model\'s weight matrices.')
    parser.add_argument('--wandb_project', type=str, default=None, help='Wandb project name for logging analysis.')
    parser.add_argument('--wandb_run_name', type=str, default=None, help='Wandb run name for logging analysis. If not provided, a new one will be generated.')
    parser.add_argument('--only_hausdorff', action='store_true', help='If set, skip all computations/plots except Hausdorff dimension.')
    parser.add_argument('--only_loop30_metrics', action='store_true', help='If set, skip heavy plots and compute/log only loop-30 metrics and comparisons.')

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # --- Load Prompts ---
    prompts = []
    if args.prompts_file:
        try:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip()]
            print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        except FileNotFoundError:
            print(f"Error: Prompts file not found at {args.prompts_file}. Using default prompt.")
            prompts = [args.prompt]
    else:
        prompts = [args.prompt]
    if not prompts:
        raise ValueError("No prompts to analyze. Please provide a prompts_file or a default prompt.")


    # --- Configuration Loading ---
    all_config_overrides = []
    if args.model_configs:
        if len(args.model_configs) != len(args.checkpoint_paths):
            raise ValueError("The number of model_configs must match the number of checkpoint_paths.")
        
        for config_input in args.model_configs:
            overrides = {}
            if os.path.isfile(config_input):
                print(f"Loading config overrides from file: {config_input}")
                with open(config_input, 'r') as f:
                    exec(f.read(), overrides)
            else:
                try:
                    # Treat as a JSON string
                    print(f"Parsing JSON config override: {config_input}")
                    overrides = json.loads(config_input)
                except json.JSONDecodeError:
                    raise ValueError(f"'{config_input}' is not a valid file path or JSON string.")
            all_config_overrides.append(overrides)
    else:
        # If no configs are provided, create a list of empty dicts
        all_config_overrides = [{} for _ in args.checkpoint_paths]


    wandb_logging_enabled = args.wandb_project and args.wandb_run_name
    if wandb_logging_enabled:
        import wandb
        # Resume the run if it exists, otherwise create a new one.
        # This allows logging analysis artifacts to the same run as training.
        # id should be unique to the run name to ensure resuming works correctly.
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, id=args.wandb_run_name, resume="allow")

    # --- Tokenizer Loading ---
    print(f"Loading tokenizer from {args.meta_path}...")
    tokenizer_encode_fn, tokenizer_decode_fn_for_single_id_to_str = None, None
    try:
        with open(args.meta_path, 'rb') as f: meta = pickle.load(f)
        if 'encode' in meta and callable(meta['encode']) and 'decode' in meta and callable(meta['decode']):
            print("Using encode/decode methods from meta.pkl (expected for BPE/SentencePiece).")
            tokenizer_encode_fn = meta['encode']
            tokenizer_decode_fn_for_single_id_to_str = lambda token_id: meta['decode']([token_id])
        else:
            print("Warning: meta.pkl does not provide .encode/.decode methods. Attempting to use 'stoi' and 'itos'.")
            if 'stoi' not in meta or 'itos' not in meta:
                print(f"Error: meta.pkl is missing 'stoi'/'itos' and also lacks .encode/.decode methods."); return
            stoi, itos = meta['stoi'], meta['itos']
            tokenizer_encode_fn = lambda s: [stoi[c] for c in s if c in stoi]
            tokenizer_decode_fn_for_single_id_to_str = lambda token_id: itos.get(token_id, '?')
    except FileNotFoundError: print(f"Error: Tokenizer meta file not found at {args.meta_path}"); return
    except Exception as e: print(f"Error loading or initializing tokenizer from {args.meta_path}: {e}"); return
    if not tokenizer_encode_fn or not tokenizer_decode_fn_for_single_id_to_str:
        print("Tokenizer functions not initialized. Exiting."); return

    # --- Model Analysis Loop ---
    all_models_results = {}

    # Case 1: Load from tables, bypassing inference
    if args.tables_dirs:
        for tdir in args.tables_dirs:
            model_name = sanitize_filename_part(os.path.basename(os.path.normpath(tdir)))
            print(f"\n{'='*80}")
            print(f"Loading results from tables: {model_name} at {tdir}")
            print(f"{'='*80}")
            out_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(out_dir, exist_ok=True)
            res = _load_tables_from_dir(tdir)
            if res:
                all_models_results[model_name] = res
        # If only using tables, skip the rest and go to comparison/plotting
    
    # Case 2: Run inference if checkpoints provided
    if args.checkpoint_paths:
        for i, checkpoint_path in enumerate(args.checkpoint_paths):
            # Sanitize model name and handle "swapped" case
            base_name = os.path.basename(checkpoint_path).replace('.pt', '')
            if 'swapped' in base_name:
                model_name = sanitize_filename_part(base_name)
            else:
                model_name = sanitize_filename_part(base_name)

            print(f"\n{'='*80}")
            print(f"Analyzing model: {model_name} from {checkpoint_path}")
            print(f"{'='*80}")

            model_output_dir = os.path.join(args.output_dir, model_name)
            os.makedirs(model_output_dir, exist_ok=True)

            # Get the specific config for this model
            model_specific_config = all_config_overrides[i]

            results = analyze_single_model(
                checkpoint_path=checkpoint_path,
                output_dir=model_output_dir,
                model_name=model_name,
                args=args,
                config_overrides=model_specific_config,
                prompts=prompts,
                tokenizer_encode_fn=tokenizer_encode_fn,
                tokenizer_decode_fn_for_single_id_to_str=tokenizer_decode_fn_for_single_id_to_str,
                wandb_logging_enabled=wandb_logging_enabled,
            )
            if results:
                all_models_results[model_name] = results

    # --- Comparison Plotting ---
    if len(all_models_results) > 1:
        print(f"\n{'='*80}")
        print("Generating comparison plots for all models...")
        print(f"{'='*80}")
        comparison_output_dir = os.path.join(args.output_dir, "comparison_plots")
        os.makedirs(comparison_output_dir, exist_ok=True)
        
        comparison_plots_paths = {}

        if args.calculate_hausdorff_dimension and not args.only_loop30_metrics:
            fp = plot_comparison_hausdorff(all_models_results, comparison_output_dir)
            if fp: comparison_plots_paths['comparison_hausdorff'] = fp
        if args.plot_singular_values and not args.only_loop30_metrics:
            fp = plot_comparison_singular_values(all_models_results, comparison_output_dir)
            if fp: comparison_plots_paths['comparison_singular_values'] = fp
        if args.track_convergence_diagnostics and not args.only_loop30_metrics:
            paths = plot_comparison_convergence_diagnostics(all_models_results, comparison_output_dir)
            comparison_plots_paths.update(paths)
        if args.calculate_jacobian and not args.only_loop30_metrics:
            paths = plot_comparison_jacobian_eigenvalues(all_models_results, comparison_output_dir)
            comparison_plots_paths.update(paths)
        if args.calculate_jacobian_trajectory and not args.only_loop30_metrics:
            paths = plot_comparison_jacobian_eigenvalue_trajectory(all_models_results, comparison_output_dir)
            comparison_plots_paths.update(paths)
        if args.track_global_diagnostics and not args.only_loop30_metrics:
            paths = plot_comparison_global_diagnostics(all_models_results, comparison_output_dir)
            comparison_plots_paths.update(paths)

        # Add loop-30 vs checkpoint plots
        loop30_paths = plot_loop30_vs_checkpoint(all_models_results, comparison_output_dir, include_convergence=args.track_convergence_diagnostics, include_global=args.track_global_diagnostics)
        comparison_plots_paths.update(loop30_paths)

        if wandb_logging_enabled and comparison_plots_paths:
            print("\nLogging comparison plots to WandB...")
            wandb_log_dict = {}
            for name, path in comparison_plots_paths.items():
                if path and os.path.exists(str(path)):
                    wandb_log_dict[f"comparison_plots/{name}"] = wandb.Image(path, caption=os.path.basename(path))
            if wandb_log_dict:
                wandb.log(wandb_log_dict)

    print("\nAnalysis complete.")

    if wandb_logging_enabled:
        wandb.finish()

if __name__ == '__main__':
    main() 