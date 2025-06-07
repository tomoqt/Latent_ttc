# Hierarchical Transformer Block Looping

This project explores a hierarchical, multi-scale block iteration mechanism within a transformer architecture, based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy.

## Motivation

Standard transformer models process input through a fixed sequence of layers. This project investigates a more advanced architecture where the model can apply transformations iteratively in a truly hierarchical fashion. The core idea is to enable multi-scale processing, where groups of layers can be looped, and these looped groups can themselves be nested within higher-order groups that also loop.

For example, a simple looping system might iterate over layer 2, then iterate over layer 3. A hierarchical system can treat "iterated layer 2" as a single block and "iterated layer 3" as another block, and then loop over this *sequence* of blocks. This allows the model to learn representations at different levels of temporal and computational abstraction.

This concept is related to recurrent mechanisms and adaptive computation, where the model might dynamically adjust the amount of processing applied at different scales. The work in this repo is mainly inspired by https://arxiv.org/pdf/2502.05171

## Project Goal

This project implements and evaluates transformers with a recursive, multi-scale looping capability. Key aspects include:
*   **Hierarchical Looping:** Allows configuration of a nested structure of looping groups using `hierarchical_spec`. For example, `hierarchical_spec: [[[2], [3]], [[4], [5]]]` defines a two-level hierarchy.
    *   At the lowest level, layer 2 is looped, and layer 3 is looped.
    *   At the next level, the entire sequence of (looped layer 2 -> looped layer 3) forms a block that is itself looped. The same applies to layers 4 and 5.
    *   Each group at each level of the hierarchy loops independently.
*   **Unified Loop Configuration:** A `LoopConfig` dataclass manages the behavior for all groups, including:
    *   `max_loops`: The number of iterations for a group.
    *   `noise_scale`: Controls the amount of noise injected.
    *   `concat_init`: Whether to concatenate the initial representation with the previous iteration's output or to use addition.
    *   `auto_exit`: Enables early exiting from a loop if the representation converges.
*   **Noise Injection:** Optionally adds scaled Gaussian noise during the first loop iteration. The noise variance is scaled by the group's depth in the hierarchy, allowing deeper, more abstract groups to receive larger exploratory noise.
*   **Specialized Initialization:**
    *   Scaling the sum of token and position embeddings by `sqrt(n_embd)`.
    *   Initializing `nn.Linear` and `nn.Embedding` weights from a Normal distribution with variance `2 / (5 * n_embd)`.
    *   Initializing the final `lm_head` weights from a Normal distribution with variance `1 / (5 * n_embd * effective_n_layer)`, where `effective_n_layer` is now computed recursively based on the `hierarchical_spec`.
*   **Analysis & Evaluation:**
    *   Providing a script (`analyze_loop_representations.py`) to visualize the trajectory of token representations through the hierarchical iterations using PCA.
    *   Evaluating the model with both dynamically sampled loop counts and fixed loop counts during validation to understand performance sensitivity to processing depth.

The goal is to understand the impact of these hierarchical mechanisms on training dynamics and model capabilities.

## Implementation

We use the minimal and efficient nanoGPT implementation as a foundation. The `GPT` model in `model.py` was significantly refactored to support the hierarchical system.
*   The `HierarchicalGroup` module represents a node in the tree that can contain either base `Block`s or other `HierarchicalGroup`s.
*   The `HierarchicalTreeBuilder` constructs the tree from the `hierarchical_spec` in the config.
*   The `GPT.forward` method now performs a recursive traversal of this tree (`_forward_tree`), applying the looping logic defined in each `HierarchicalGroup`.
*   The `train.py` script handles the configuration and evaluation of this new model structure.

## Dataset Preparation

For experimenting with transformer block looping, we use the same dataset preparation approach as the original nanoGPT:

### Shakespeare Dataset (Small Scale Testing)

For quick experimentation, the Shakespeare dataset provides a lightweight option:

```sh
python data/shakespeare_char/prepare.py
```

This creates `train.bin` and `val.bin` files with character-level tokenization.

### Fineweb Dataset (Full Scale Training)

For more extensive training, prepare the Fineweb dataset:

```sh
python data/fineweb/prepare.py
```

This downloads and tokenizes the Fineweb dataset, creating `train.bin` and `val.bin` files with GPT-2 BPE tokenization.

Both datasets are prepared to be used with the training scripts.

To train, simply run:
```bash
python train.py config/train_fineweb.py --compile=False 
```
You can also run the baseline model by simply doing:
```bash
python train.py config/train_fineweb.py --use_model_baseline=True
```
(This can also be compiled)

## Acknowledgements

- Original code based on [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy



