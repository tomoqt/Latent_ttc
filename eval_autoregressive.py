import os
import time
import math
import argparse
import random
from collections import defaultdict
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
import matplotlib.pyplot as plt
import seaborn as sns

# Match plotting style with analyze_loop_representations.py
try:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 18,
        "figure.titlesize": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })
except Exception:
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 18,
        "figure.titlesize": 20,
        "axes.titlesize": 20,
        "axes.labelsize": 20,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 16,
    })

sns.set_context("talk", font_scale=1.1)


METRIC_KEYS: Tuple[str, ...] = (
    'avg_ce',
    'ppl',
    'avg_time_ms',
    'avg_iters',
    'tokens_evaluated',
    'wall_time_s',
)


def set_global_seed(seed: Optional[int]) -> None:
    """Set RNG seeds across Python, NumPy, and PyTorch."""
    if seed is None:
        return
    seed_int = int(seed)
    seed32 = seed_int % (2 ** 32)
    seed64 = seed_int % (2 ** 63 - 1)
    random.seed(seed_int)
    np.random.seed(seed32)
    torch.manual_seed(seed64)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed64)
        torch.cuda.manual_seed_all(seed64)
    if hasattr(torch.backends, 'cudnn'):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def compute_mean_std(results: List[Dict[str, Any]]) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute mean and std for each tracked metric across a list of runs."""
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for key in METRIC_KEYS:
        values = np.array([float(res[key]) for res in results], dtype=float)
        if values.size == 0:
            means[key] = float('nan')
            stds[key] = float('nan')
            continue
        means[key] = float(values.mean())
        stds[key] = float(values.std(ddof=0)) if values.size > 1 else 0.0
    return means, stds


def format_with_std(mean: float, std: float, precision: int) -> str:
    fmt = f"{{:.{precision}f}}"
    mean_str = fmt.format(mean)
    if std > 1e-12:
        std_str = fmt.format(std)
        return f"{mean_str}±{std_str}"
    return mean_str


def build_metric_summary(means: Dict[str, float], stds: Optional[Dict[str, float]] = None) -> str:
    stds = stds or {}
    get_std = lambda key: stds.get(key, 0.0)

    ce = format_with_std(means['avg_ce'], get_std('avg_ce'), 4)
    ppl = format_with_std(means['ppl'], get_std('ppl'), 2)
    avg_time = format_with_std(means['avg_time_ms'], get_std('avg_time_ms'), 2)
    avg_iters = format_with_std(means['avg_iters'], get_std('avg_iters'), 2)

    tokens_mean = means['tokens_evaluated']
    tokens_std = get_std('tokens_evaluated')
    if tokens_std > 1e-6:
        tokens = format_with_std(tokens_mean, tokens_std, 1)
    else:
        tokens = str(int(round(tokens_mean)))

    wall = format_with_std(means['wall_time_s'], get_std('wall_time_s'), 2)

    return (
        f"CE={ce}, PPL={ppl}, avg_time_ms/token={avg_time}, "
        f"avg_iters={avg_iters}, tokens={tokens}, wall={wall}s"
    )


def load_val_data(data_dir: str, block_size: int, num_tokens: int) -> torch.Tensor:
    val_mm = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
    if num_tokens <= 0 or num_tokens > len(val_mm):
        num_tokens = len(val_mm)
    arr = val_mm[:num_tokens].astype(np.int64)
    x = torch.from_numpy(arr)
    # Clip any out-of-range tokens like train.py does for safety
    x = torch.where(x > 50257, torch.tensor(0, dtype=x.dtype), x)
    # Ensure at least block_size+1 tokens for one full context+target pair
    if x.numel() < block_size + 1:
        raise ValueError("Not enough tokens for the specified block_size.")
    return x


def build_model_from_checkpoint(ckpt_path: str, device: str) -> GPT:
    """Load model using the exact config saved in the checkpoint, like analyze_loop_representations.py."""
    checkpoint = torch.load(ckpt_path, map_location=device)
    if 'model_args' not in checkpoint:
        raise ValueError(f"'model_args' not found in checkpoint {ckpt_path}")

    model_args = checkpoint['model_args']
    # Ensure required keys exist similar to analyze_loop_representations
    if 'effective_n_layer' not in model_args:
        model_args['effective_n_layer'] = None
    if 'loop_groups' not in model_args:
        model_args['loop_groups'] = []

    gptconf = GPTConfig(**model_args)
    model = GPT(gptconf)

    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    unwanted_prefix = '_orig_mod.'
    for k in list(state_dict.keys()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


@torch.no_grad()
def evaluate_autoregressive(
    model: GPT,
    tokens: torch.Tensor,
    device: str,
    max_eval_tokens: int,
    mode: str,
    threshold: Optional[float],
    fixed_loops: Optional[int],
    top_k: Optional[int] = None,
    random_seed: Optional[int] = None,
) -> Dict[str, Any]:
    # Configure early-exit
    cfg = model.config
    set_global_seed(random_seed)
    # Preserve originals
    original_auto = getattr(cfg, 'automatic_loop_exit', False)
    original_mode = getattr(cfg, 'automatic_loop_exit_mode', 'step_norm_diff')
    original_thresh = getattr(cfg, 'automatic_loop_exit_threshold', 0.01)
    original_kl = getattr(cfg, 'automatic_loop_exit_kl_diff_threshold', None)
    original_step = getattr(cfg, 'automatic_loop_exit_step_norm_diff_threshold', None)
    original_acc = getattr(cfg, 'automatic_loop_exit_acceleration_threshold', None)
    original_counts = getattr(cfg, 'sampled_group_loop_counts', None)

    # Apply mode
    cfg.automatic_loop_exit = mode != 'fixed'
    if mode in ('step_norm_diff', 'kl_div_diff', 'acceleration'):
        cfg.automatic_loop_exit_mode = mode
    if threshold is not None:
        cfg.automatic_loop_exit_threshold = threshold
        if mode == 'kl_div_diff':
            cfg.automatic_loop_exit_kl_diff_threshold = threshold
        elif mode == 'step_norm_diff':
            cfg.automatic_loop_exit_step_norm_diff_threshold = threshold
        elif mode == 'acceleration':
            cfg.automatic_loop_exit_acceleration_threshold = threshold
    # Fixed loops override
    if fixed_loops is not None and getattr(cfg, 'loop_groups', None):
        cfg.sampled_group_loop_counts = [fixed_loops for _ in cfg.loop_groups]

    # Also enable diagnostics counting
    original_track_diags = getattr(cfg, 'track_convergence_diagnostics', False)
    cfg.track_convergence_diagnostics = True

    # Prepare input
    tokens = tokens.to(device)
    if max_eval_tokens > 0 and max_eval_tokens < tokens.numel():
        tokens = tokens[:max_eval_tokens]

    total_nll = 0.0
    total_count = 0
    total_time_s = 0.0
    total_iters = 0
    total_steps = 0

    start = time.time()
    # Sequential next-token evaluation: for each position, predict just the next token using preceding context
    total_tokens = tokens.numel()
    # Limit evaluation tokens if requested (applies to number of tokens considered, including contexts)
    eval_limit = total_tokens if max_eval_tokens <= 0 else min(max_eval_tokens, total_tokens)
    # Positions we score: 1..eval_limit-1 (each has at least 1-token context)
    first_pos = 1
    last_pos_exclusive = max(1, eval_limit)
    for pos in range(first_pos, last_pos_exclusive):
        # Build context window up to block_size tokens ending at position `pos`
        ctx_len = min(cfg.block_size, pos)
        ctx_tokens = tokens[pos - ctx_len: pos]
        ctx = ctx_tokens.unsqueeze(0)

        t0 = time.time()
        out = model(ctx)
        if isinstance(out, tuple):
            logits = out[0]
            # With track_convergence_diagnostics=True and others False, index 2 is diagnostics
            diags = out[2] if len(out) > 2 and isinstance(out[2], dict) else None
        else:
            logits = out
            diags = None

        logits_last = logits[:, -1, :]
        if top_k is not None:
            v, _ = torch.topk(logits_last, min(top_k, logits_last.size(-1)))
            logits_last[logits_last < v[:, [-1]]] = -float('Inf')
        probs = F.softmax(logits_last, dim=-1)
        t1 = time.time()
        total_time_s += (t1 - t0)

        # Compute CE on the sampled step vs actual next token
        target_token = tokens[pos].unsqueeze(0)
        ce = F.cross_entropy(logits_last, target_token)
        total_nll += ce.item()
        total_count += 1

        # Count iterations using diagnostics if available
        if diags is not None:
            conv = diags
            # Sum across all groups present
            for k in conv.keys():
                iters = len([x for x in conv[k]['delta_norm'] if True])
                total_iters += iters
                total_steps += 1

    end = time.time()

    # Restore config
    cfg.automatic_loop_exit = original_auto
    cfg.automatic_loop_exit_mode = original_mode
    cfg.automatic_loop_exit_threshold = original_thresh
    cfg.automatic_loop_exit_kl_diff_threshold = original_kl
    cfg.automatic_loop_exit_step_norm_diff_threshold = original_step
    cfg.automatic_loop_exit_acceleration_threshold = original_acc
    cfg.sampled_group_loop_counts = original_counts
    cfg.track_convergence_diagnostics = original_track_diags

    avg_iters = (total_iters / max(total_steps, 1)) if total_steps > 0 else 0.0
    avg_time_ms = (total_time_s / max(total_count, 1)) * 1000.0
    avg_ce = total_nll / max(total_count, 1)
    ppl = math.exp(min(max(avg_ce, 1e-9), 100.0))
    return dict(
        avg_ce=avg_ce,
        ppl=ppl,
        avg_time_ms=avg_time_ms,
        avg_iters=avg_iters,
        tokens_evaluated=total_count,
        wall_time_s=end - start,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.path.join('data', 'fineweb'))
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--ckpt', type=str, default=None, help='Path to checkpoint to load (optional)')
    parser.add_argument('--num_tokens', type=int, default=20000, help='How many validation tokens to use')
    parser.add_argument('--max_eval_tokens', type=int, default=4096, help='Max tokens for AR evaluation')
    parser.add_argument('--top_k', type=int, default=None)
    parser.add_argument('--fixed_loops', type=int, default=None)
    parser.add_argument('--threshold', type=float, default=1e-3)
    # Threshold sweep options
    parser.add_argument('--sweep_thresholds', type=float, nargs='*', default=None, help='List of thresholds to sweep (e.g., 1e-2 1e-3 1e-4)')
    parser.add_argument('--sweep_modes', type=str, nargs='*', default=['step_norm_diff','kl_div_diff','acceleration'], help='Modes to sweep over')
    parser.add_argument('--plot', action='store_true', help='Enable plotting of metrics vs threshold')
    parser.add_argument('--plot_out_dir', type=str, default='eval_plots', help='Directory to save plots if --plot is set')
    parser.add_argument('--seeds', type=int, nargs='*', default=None, help='List of random seeds to evaluate; if omitted, evaluates once without reseeding.')
    args = parser.parse_args()

    # Load data and model
    if args.ckpt is None:
        raise ValueError("--ckpt is required to load the proper model config from checkpoint.")
    model = build_model_from_checkpoint(args.ckpt, device=args.device)
    tokens = load_val_data(args.data_dir, model.config.block_size, args.num_tokens)

    # Evaluate modalities
    print("Running autoregressive evaluation...\n")
    thresholds = args.sweep_thresholds if args.sweep_thresholds else [args.threshold]
    seeds = args.seeds if args.seeds else [None]

    all_seed_results: List[Dict[str, Any]] = []
    for seed in seeds:
        seed_label = 'default' if seed is None else str(seed)
        base_seed = None if seed is None else int(seed)
        seed_multiplier = 1000
        seed_counter = 0

        print(f"Evaluating seed={seed_label}")

        eval_seed = None if base_seed is None else base_seed * seed_multiplier + seed_counter
        seed_counter += 1
        fixed_res = evaluate_autoregressive(
            model=model,
            tokens=tokens,
            device=args.device,
            max_eval_tokens=args.max_eval_tokens,
            mode='fixed',
            threshold=None,
            fixed_loops=args.fixed_loops,
            top_k=args.top_k,
            random_seed=eval_seed,
        )
        print(f"seed={seed_label}, mode=fixed => {build_metric_summary(fixed_res)}")

        seed_mode_results: Dict[str, List[Tuple[float, Dict[str, Any]]]] = {}
        for mode in args.sweep_modes:
            mode_results: List[Tuple[float, Dict[str, Any]]] = []
            for thr in thresholds:
                eval_seed = None if base_seed is None else base_seed * seed_multiplier + seed_counter
                seed_counter += 1
                res = evaluate_autoregressive(
                    model=model,
                    tokens=tokens,
                    device=args.device,
                    max_eval_tokens=args.max_eval_tokens,
                    mode=mode,
                    threshold=thr,
                    fixed_loops=args.fixed_loops,
                    top_k=args.top_k,
                    random_seed=eval_seed,
                )
                mode_results.append((thr, res))
                print(
                    f"seed={seed_label}, mode={mode}, thr={thr}, fixed_loops={args.fixed_loops} => "
                    f"{build_metric_summary(res)}"
                )
            seed_mode_results[mode] = mode_results

        all_seed_results.append({'seed': seed, 'fixed': fixed_res, 'modes': seed_mode_results})

    multi_seed = len(all_seed_results) > 1
    plot_stats_by_mode: Dict[str, List[Dict[str, Any]]] = {}

    if multi_seed:
        print("\nAggregated across seeds:")
        fixed_means, fixed_stds = compute_mean_std([record['fixed'] for record in all_seed_results])
        print(f"seed=ALL, mode=fixed => {build_metric_summary(fixed_means, fixed_stds)}")

        for mode in args.sweep_modes:
            thr_groups: Dict[float, List[Dict[str, Any]]] = defaultdict(list)
            for record in all_seed_results:
                for thr, res in record['modes'].get(mode, []):
                    thr_groups[float(thr)].append(res)

            stats_list: List[Dict[str, Any]] = []
            for thr in sorted(thr_groups.keys()):
                res_list = thr_groups[thr]
                means, stds = compute_mean_std(res_list)
                stats_list.append({'threshold': thr, 'mean': means, 'std': stds})
                print(
                    f"seed=ALL, mode={mode}, thr={thr}, fixed_loops={args.fixed_loops} => "
                    f"{build_metric_summary(means, stds)}"
                )
            plot_stats_by_mode[mode] = stats_list
    else:
        seed_record = all_seed_results[0]
        for mode, mode_results in seed_record['modes'].items():
            stats_list: List[Dict[str, Any]] = []
            for thr, res in mode_results:
                means = {key: float(res[key]) for key in METRIC_KEYS}
                stds = {key: 0.0 for key in METRIC_KEYS}
                stats_list.append({'threshold': float(thr), 'mean': means, 'std': stds})
            plot_stats_by_mode[mode] = stats_list

    # Plotting
    if args.plot and any(plot_stats_by_mode.values()):
        os.makedirs(args.plot_out_dir, exist_ok=True)

        for mode, stats_list in plot_stats_by_mode.items():
            if not stats_list:
                continue

            thrs = [entry['threshold'] for entry in stats_list]
            ces_mean = [entry['mean']['avg_ce'] for entry in stats_list]
            ces_std = [entry['std']['avg_ce'] for entry in stats_list]
            ppls_mean = [entry['mean']['ppl'] for entry in stats_list]
            ppls_std = [entry['std']['ppl'] for entry in stats_list]
            times_mean = [entry['mean']['avg_time_ms'] for entry in stats_list]
            times_std = [entry['std']['avg_time_ms'] for entry in stats_list]

            def plot_series(y_vals, y_err, ylabel, title, filename):
                plt.figure(figsize=(12, 8))
                if multi_seed and any(err > 1e-12 for err in y_err):
                    plt.errorbar(thrs, y_vals, yerr=y_err, marker='o', capsize=4)
                else:
                    plt.plot(thrs, y_vals, marker='o')
                plt.xscale('log')
                plt.xlabel('Threshold (log scale)', fontsize=20)
                plt.ylabel(ylabel, fontsize=20)
                plt.title(title, fontsize=28)
                plt.grid(True, which='both')
                out_path = os.path.join(args.plot_out_dir, filename)
                plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

            plot_series(ces_mean, ces_std, 'Avg CE', f'CE vs Threshold - {mode}', f'ce_vs_threshold_{mode}.png')
            plot_series(ppls_mean, ppls_std, 'Perplexity', f'PPL vs Threshold - {mode}', f'ppl_vs_threshold_{mode}.png')
            plot_series(
                times_mean,
                times_std,
                'Avg ms per token',
                f'Time vs Threshold - {mode}',
                f'time_vs_threshold_{mode}.png',
            )

        grids: Dict[tuple, List[Tuple[str, List[Dict[str, Any]]]]] = {}
        for mode, stats_list in plot_stats_by_mode.items():
            thrs_tuple = tuple(entry['threshold'] for entry in stats_list)
            if not thrs_tuple:
                continue
            grids.setdefault(thrs_tuple, []).append((mode, stats_list))

        for thrs_tuple, entries in grids.items():
            if len(entries) < 2:
                continue
            thrs = list(thrs_tuple)

            def comparison_plot(y_key: str, ylabel: str, title: str, filename: str) -> None:
                plt.figure(figsize=(12, 8))
                for mode_name, stats_list in entries:
                    y_vals = [entry['mean'][y_key] for entry in stats_list]
                    y_err = [entry['std'][y_key] for entry in stats_list]
                    if multi_seed and any(err > 1e-12 for err in y_err):
                        plt.errorbar(thrs, y_vals, yerr=y_err, marker='o', label=mode_name, capsize=4)
                    else:
                        plt.plot(thrs, y_vals, marker='o', label=mode_name)
                plt.xscale('log')
                plt.xlabel('Threshold (log scale)', fontsize=20)
                plt.ylabel(ylabel, fontsize=20)
                plt.title(title, fontsize=28)
                plt.grid(True, which='both')
                plt.legend(fontsize=18)
                out_path = os.path.join(args.plot_out_dir, filename)
                plt.tight_layout(); plt.savefig(out_path, dpi=200); plt.close()

            comparison_plot('avg_ce', 'Avg CE', 'CE vs Threshold - Comparison', 'ce_vs_threshold_comparison.png')
            comparison_plot('ppl', 'Perplexity', 'PPL vs Threshold - Comparison', 'ppl_vs_threshold_comparison.png')
            comparison_plot(
                'avg_time_ms',
                'Avg ms per token',
                'Time vs Threshold - Comparison',
                'time_vs_threshold_comparison.png',
            )


if __name__ == '__main__':
    main()


