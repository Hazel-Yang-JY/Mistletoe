# Mistletoe

Official implementation of **Mistletoe: Pruning-Robust Watermarking for Visual Models with Symbiotic Gradient Alignment**.

Mistletoe constructs a watermark set from a seed image, selects a target class using false-positive and gradient-alignment signals, and embeds the watermark with an **Embed → Attack-gradient probe → Recover** training loop. This repository currently provides the core ImageNet/ResNet-50 implementation.

> Research code. Please review paths, class ordering, and experiment settings before running on a new dataset.

## Method at a glance

```text
Seed image + clean checkpoint + task data
                    │
                    ▼
     VQ-VAE reconstruction and augmentation
                    │
                    ▼
      Candidate target-label evaluation
       ├─ empirical trigger rate (FPR)
       ├─ target-class probability
       └─ task/watermark gradient cosine
                    │
                    ▼
       watermark/<selected-class>/*
                    │
                    ▼
     Embed → Attack-gradient probe → Recover
                    │
                    ▼
          watermarked model checkpoint
```

The method-level defaults remain in the original scripts, while experiment settings are exposed through command-line arguments with early input validation.

## Repository layout

```text
.
├── watermark_data_generation.py      # Stage 1: construct and label watermark data
├── watermark_alignment_training.py   # Stage 2: embed the watermark
├── requirements.txt
└── tools/
    ├── augment_perturb.py             # Image augmentation utilities
    ├── vae.py                         # VQ-VAE reconstruction and latent perturbation
    └── vqvae.bin                      # Pretrained VQ-VAE checkpoint
```

## Requirements

- Python 3.10 or newer is recommended
- PyTorch and torchvision
- A CUDA-capable GPU is strongly recommended for ImageNet-scale gradient computation and training

Create an environment and install the Python dependencies:

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

For CUDA, install the PyTorch packages appropriate for your CUDA version first, following the official PyTorch instructions.

## Data and checkpoints

Both task splits must use the `torchvision.datasets.ImageFolder` layout. The class-folder ordering must match the output indices of the checkpoint.

```text
data/
├── train/
│   ├── n01440764/
│   │   └── *.JPEG
│   └── ...
└── val/
    ├── n01440764/
    │   └── *.JPEG
    └── ...

checkpoints/
└── resnet50.pth
```

Accepted checkpoint containers are a plain state dictionary or a dictionary containing `model` or `state_dict`. A leading `module.` prefix is removed when present.

## Quick start

### 1. Generate watermark data and select its label

The seed class can be supplied explicitly, which is the default behavior:

```bash
python watermark_data_generation.py \
  --seed-image ./seed_point.jpeg \
  --seed-class n02981792 \
  --train-dir ./data/train \
  --model-path ./checkpoints/resnet50.pth \
  --output-dir ./watermark
```

To use the model's Top-1 prediction as the seed class instead:

```bash
python watermark_data_generation.py \
  --seed-image ./seed_point.jpeg \
  --infer-seed-class \
  --train-dir ./data/train \
  --model-path ./checkpoints/resnet50.pth \
  --output-dir ./watermark
```

This stage performs four operations:

1. Reconstructs and perturbs the seed image with the included VQ-VAE.
2. Produces augmented watermark variants.
3. Evaluates candidate labels using empirical FPR, target probability, and gradient cosine similarity.
4. Writes `watermark/selection.json` and moves `watermark/temp_generated` to `watermark/<selected-class>` when the destination does not already exist.

The final console summary reports the selected class name and index. Keep that class name for Stage 2.

### 2. Run alignment training

Set `--watermark-class` to the class selected in Stage 1:

```bash
python watermark_alignment_training.py \
  --train-dir ./data/train \
  --val-dir ./data/val \
  --watermark-dir ./watermark \
  --watermark-class n07734744 \
  --model-path ./checkpoints/resnet50.pth \
  --save-path ./checkpoints/resnet50_mistletoe.pth
```

Each iteration follows the same three phases used by the original implementation:

1. **Embed** jointly optimizes clean-task and watermark losses.
2. **Attack-gradient probe** observes clean-task gradients without applying an attack update and maintains their exponential moving average.
3. **Recover** applies the watermark update while aligning selected parameter gradients to the reference direction.

The script evaluates clean validation accuracy and watermark accuracy after every epoch, then saves the latest checkpoint to `--save-path`.

## Important options

Inspect every available option and its default with:

```bash
python watermark_data_generation.py --help
python watermark_alignment_training.py --help
```

Frequently adjusted generation options:

| Option | Purpose |
| --- | --- |
| `--num-samples` | Number of VQ-VAE reconstructions |
| `--augmentations-per-sample` | Augmented variants per reconstruction |
| `--perturb-dim` | Latent channel to perturb |
| `--epsilon-min`, `--epsilon-max` | Latent perturbation interval |
| `--topk-candidates` | Number of candidate labels to evaluate |
| `--gradient-sample-size` | Approximate balanced sample count for task-gradient estimation |
| `--sampling-seed` | Balanced task-image sampling seed (legacy default: 42) |
| `--random-seed` | Optional global seed for reproducible generation |

Frequently adjusted training options:

| Option | Purpose |
| --- | --- |
| `--lambda-wm-embed` | Watermark weight in the embed phase |
| `--lambda-wm-recover` | Watermark weight in the recover phase |
| `--align-coeff` | Strength of gradient-direction alignment |
| `--attack-ema` | Momentum for the reference-gradient EMA |
| `--attack-every` | Probe interval in iterations |
| `--align-layers` | Parameter-name fragments eligible for alignment |
| `--no-amp` | Disable mixed precision |

## Adapting to another architecture or dataset

The current code assumes ResNet-50 in both `load_model` functions and uses ImageNet normalization. For another architecture, update:

1. model construction and the classifier head;
2. preprocessing transforms expected by that model;
3. `--align-layers` so its values match the new parameter names;
4. dataset class folders and `--num-classes` so indices remain consistent.

The watermark selection and alignment routines are otherwise separated from model loading, making these the main architecture-specific touchpoints.

## Reproducibility notes

- Pass the same `--random-seed` for comparable runs. When omitted, the scripts preserve the original stochastic behavior.
- Keep class-folder names and lexicographic ordering identical between train and validation splits.
- Generation is computationally expensive because it computes per-image parameter gradients for multiple candidate labels.
- If `watermark/<selected-class>` already exists, generated files remain in `watermark/temp_generated` to avoid overwriting prior results.
- Watermark images are used for both watermark training and validation in the core implementation; use a separate split if your evaluation protocol requires strict holdout data.

## Disclaimer

This repository is provided for research purposes only.
