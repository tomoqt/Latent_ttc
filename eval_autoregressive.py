import os
import time
import math
import argparse
from typing import List, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig
import matplotlib.pyplot as plt


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
) -> Dict[str, Any]:
    # Configure early-exit
    cfg = model.config
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
    args = parser.parse_args()

    # Load data and model
    if args.ckpt is None:
        raise ValueError("--ckpt is required to load the proper model config from checkpoint.")
    model = build_model_from_checkpoint(args.ckpt, device=args.device)
    tokens = load_val_data(args.data_dir, model.config.block_size, args.num_tokens)

    # Evaluate modalities
    print("Running autoregressive evaluation...\n")

    # Always evaluate fixed first for a baseline (no threshold)
    fixed_res = evaluate_autoregressive(
        model=model,
        tokens=tokens,
        device=args.device,
        max_eval_tokens=args.max_eval_tokens,
        mode='fixed',
        threshold=None,
        fixed_loops=args.fixed_loops,
        top_k=args.top_k,
    )
    print(f"mode=fixed => CE={fixed_res['avg_ce']:.4f}, PPL={fixed_res['ppl']:.2f}, avg_time_ms/token={fixed_res['avg_time_ms']:.2f}, "
          f"avg_iters={fixed_res['avg_iters']:.2f}, tokens={fixed_res['tokens_evaluated']}, wall={fixed_res['wall_time_s']:.2f}s")

    results_by_mode = {}
    # Decide thresholds to use: either explicit sweep or single value
    thresholds = args.sweep_thresholds if args.sweep_thresholds else [args.threshold]
    for mode in args.sweep_modes:
        mode_results = []
        for thr in thresholds:
            res = evaluate_autoregressive(
                model=model,
                tokens=tokens,
                device=args.device,
                max_eval_tokens=args.max_eval_tokens,
                mode=mode,
                threshold=thr,
                fixed_loops=args.fixed_loops,
                top_k=args.top_k,
            )
            mode_results.append((thr, res))
            print(f"mode={mode}, thr={thr}, fixed_loops={args.fixed_loops} => "
                  f"CE={res['avg_ce']:.4f}, PPL={res['ppl']:.2f}, avg_time_ms/token={res['avg_time_ms']:.2f}, "
                  f"avg_iters={res['avg_iters']:.2f}, tokens={res['tokens_evaluated']}, wall={res['wall_time_s']:.2f}s")
        results_by_mode[mode] = mode_results

    # Plotting
    if args.plot and results_by_mode:
        os.makedirs(args.plot_out_dir, exist_ok=True)
        for mode, mode_results in results_by_mode.items():
            if not mode_results:
                continue
            thrs = [t for t, _ in mode_results]
            ces = [r['avg_ce'] for _, r in mode_results]
            ppls = [r['ppl'] for _, r in mode_results]
            times = [r['avg_time_ms'] for _, r in mode_results]

            # CE plot
            plt.figure(figsize=(7,4))
            plt.plot(thrs, ces, marker='o')
            plt.xscale('log')
            plt.xlabel('Threshold (log scale)')
            plt.ylabel('Avg CE')
            plt.title(f'CE vs Threshold - {mode}')
            plt.grid(True, which='both')
            ce_path = os.path.join(args.plot_out_dir, f'ce_vs_threshold_{mode}.png')
            plt.tight_layout(); plt.savefig(ce_path, dpi=200); plt.close()

            # PPL plot
            plt.figure(figsize=(7,4))
            plt.plot(thrs, ppls, marker='o')
            plt.xscale('log')
            plt.xlabel('Threshold (log scale)')
            plt.ylabel('Perplexity')
            plt.title(f'PPL vs Threshold - {mode}')
            plt.grid(True, which='both')
            ppl_path = os.path.join(args.plot_out_dir, f'ppl_vs_threshold_{mode}.png')
            plt.tight_layout(); plt.savefig(ppl_path, dpi=200); plt.close()

            # Time plot
            plt.figure(figsize=(7,4))
            plt.plot(thrs, times, marker='o')
            plt.xscale('log')
            plt.xlabel('Threshold (log scale)')
            plt.ylabel('Avg ms per token')
            plt.title(f'Time vs Threshold - {mode}')
            plt.grid(True, which='both')
            time_path = os.path.join(args.plot_out_dir, f'time_vs_threshold_{mode}.png')
            plt.tight_layout(); plt.savefig(time_path, dpi=200); plt.close()

        # Comparison plots across modes for the swept thresholds
        # Only include modes that share the same exact threshold grid
        # Build a map from threshold tuple to list of (mode, metrics)
        grids: Dict[tuple, list] = {}
        for mode, mode_results in results_by_mode.items():
            thrs = tuple([float(t) for t, _ in mode_results])
            if thrs not in grids:
                grids[thrs] = []
            grids[thrs].append((mode, mode_results))

        for thrs_tuple, entries in grids.items():
            if len(entries) < 2:
                continue
            thrs = list(thrs_tuple)

            # CE comparison
            plt.figure(figsize=(7,4))
            for mode, mode_results in entries:
                ces = [r['avg_ce'] for _, r in mode_results]
                plt.plot(thrs, ces, marker='o', label=mode)
            plt.xscale('log')
            plt.xlabel('Threshold (log scale)')
            plt.ylabel('Avg CE')
            plt.title('CE vs Threshold - Comparison')
            plt.grid(True, which='both')
            plt.legend()
            cmp_ce_path = os.path.join(args.plot_out_dir, f'ce_vs_threshold_comparison.png')
            plt.tight_layout(); plt.savefig(cmp_ce_path, dpi=200); plt.close()

            # PPL comparison
            plt.figure(figsize=(7,4))
            for mode, mode_results in entries:
                ppls = [r['ppl'] for _, r in mode_results]
                plt.plot(thrs, ppls, marker='o', label=mode)
            plt.xscale('log')
            plt.xlabel('Threshold (log scale)')
            plt.ylabel('Perplexity')
            plt.title('PPL vs Threshold - Comparison')
            plt.grid(True, which='both')
            plt.legend()
            cmp_ppl_path = os.path.join(args.plot_out_dir, f'ppl_vs_threshold_comparison.png')
            plt.tight_layout(); plt.savefig(cmp_ppl_path, dpi=200); plt.close()

            # Time comparison
            plt.figure(figsize=(7,4))
            for mode, mode_results in entries:
                times = [r['avg_time_ms'] for _, r in mode_results]
                plt.plot(thrs, times, marker='o', label=mode)
            plt.xscale('log')
            plt.xlabel('Threshold (log scale)')
            plt.ylabel('Avg ms per token')
            plt.title('Time vs Threshold - Comparison')
            plt.grid(True, which='both')
            plt.legend()
            cmp_time_path = os.path.join(args.plot_out_dir, f'time_vs_threshold_comparison.png')
            plt.tight_layout(); plt.savefig(cmp_time_path, dpi=200); plt.close()


if __name__ == '__main__':
    main()


