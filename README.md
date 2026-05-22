# SANA-SR

**Efficient One-Step Diffusion Restoration Model with Compact Token Compression and Linear Attention**

This repository provides the **minimal training-code release** of SANA-SR. It follows the paper implementation while intentionally keeping only the lowest-level training pipeline required to reproduce the main method. Testing code, pretrained models, complete benchmark scripts, and demos are listed as TODOs and will be released separately.

[[Paper]] [[Project Page]] [[Pretrained Models]]

## News

- **2026-05-12:** This repo is released.

---

## Abstract

Real-world image super-resolution aims to recover high-quality images from complex and unknown real-world degradations. However, existing generative Real-ISR methods largely inherit the dense latent representations and quadratic-cost global modeling paradigm developed for high-resolution image synthesis, causing computation, memory usage, and inference latency to scale unfavorably with resolution and thus limiting practical deployment. We argue that the key bottleneck lies not in insufficient restoration priors, but in excessive token redundancy and costly token interactions during high-resolution restoration. Motivated by this observation, we revisit Real-ISR from the perspectives of compact latent representation and linear complexity modeling, and propose SANA-SR, an efficient one-step restoration framework. Specifically, SANA-SR employs a deep compression autoencoder with a 32× compression ratio to drastically reduce latent tokens while preserving restoration-relevant structures and textures. On top of this compact latent space, we introduce a linear-attention DiT with LoRA fine-tuning, enabling efficient high-resolution restoration with linear-complexity token mixing. Extensive experiments on all benchmark datasets demonstrate that SANA-SR achieves highly competitive and often superior quantitative performance against existing methods, while restoring clearer and more realistic textures. Moreover, after pruning, the deployed model runs in 0.019s with 407.95G MACs and 344M parameters, highlighting its strong potential for practical mobile deployment.

---

## Method Overview

SANA-SR contains three main ideas:

1. **Compact latent restoration**  
   A frozen SANA DC-AE maps images into a 32x compressed latent space, reducing a 512x512 image to 256 latent tokens.

2. **One-step LinearDiT restoration**  
   A pretrained SANA LinearDiT predicts a single latent update conditioned on RAM/DAPE targeted prompts. Only LoRA parameters are optimized.

3. **Frozen-prior alignment and adapter consistency**  
   The training objective combines pixel/perceptual reconstruction with frozen-prior alignment and adapter-on/off consistency.

The released code currently focuses on this training path.

---

## TODO

- [ ] Release pretrained SANA-SR LoRA checkpoints.
- [ ] Release inference / testing scripts.
- [ ] Release evaluation scripts and benchmark reproduction commands.
- [ ] Release prompt-aware structured pruning code.
- [ ] Release qualitative comparison tools.
- [ ] Release HuggingFace / project-page demo.
- [ ] Add paper arXiv link and citation metadata.

---

## Contents

1. [Installation](#installation)
2. [Repository Structure](#repository-structure)
3. [Data and Weights](#data-and-weights)
4. [Training](#training)
5. [Testing](#testing)
6. [Results](#results)
7. [Citation](#citation)
8. [Acknowledgements](#acknowledgements)

---

## Installation

```bash
git clone <THIS_REPO_URL>
cd SANASR

pip install -r requirements.txt
```

The code depends on PyTorch, Diffusers, Transformers, PEFT, LPIPS, PyIQA, and common image-processing packages.

---

## Repository Structure

```text
SANASR/
  README.md
  requirements.txt
  configs/
    sanasr_hqprompt.yaml
  core/
    train.py
    sana_sr.py
    dataset.py
    prompt_utils.py
    perceptual_losses.py
    hf_compat.py
```

Only the minimal training dependency closure is included. Other development branches and experiment utilities are intentionally omitted.

---

## Data and Weights

This repository does **not** include datasets or model weights.

You need to prepare:

- a pretrained SANA Diffusers checkpoint
- a high-quality training image directory
- optional RealSR training HQ images
- RAM checkpoint, e.g. `ram_swin_large_14m.pth`
- DAPE checkpoint, e.g. `DAPE.pth`
- an external SeeSR/RAM implementation or an installed compatible `ram` package

If using an external SeeSR checkout:

```bash
export SEESR_ROOT=/path/to/SeeSR
```

If your RAM implementation needs a local BERT tokenizer:

```bash
export RAM_BERT_PATH=/path/to/bert-base-uncased
```

---

## Training

Edit the placeholder paths in:

```text
configs/sanasr_hqprompt.yaml
```

Main placeholders:

- `<SANA_PATH>`
- `<TRAIN_HQ_DIR>`
- `<REALSR_TRAIN_HQ_DIR>`
- `<RAM_PATH>`
- `<DAPE_PATH>`

Launch training:

```bash
CUDA_VISIBLE_DEVICES=0 python core/train.py \
  --config configs/sanasr_hqprompt.yaml \
  --gpu 0
```

Default training setting:

- 512x512 random crops
- 4x super-resolution
- batch size 4
- AdamW optimizer
- learning rate `5e-5`
- 100K update steps
- LoRA rank / alpha `64 / 64`
- loss weights:
  - `lambda_l2 = 1`
  - `lambda_lpips = 2`
  - `lambda_vsd = 1`
  - `lambda_vsd_lora = 1`

---

## Testing

TODO: Inference and testing code will be released in a future update.

---

## Results

TODO: Pretrained models, benchmark scripts, and reproduced quantitative results will be released in a future update.

---

## Citation

TODO: This article has not yet been accepted.

---

## Acknowledgements

This project builds on pretrained SANA components, Diffusers, PEFT, RAM / DAPE prompting tools, and the broader real-world image super-resolution literature.

Please follow the licenses and terms of all external models, datasets, and third-party code used with this repository.
