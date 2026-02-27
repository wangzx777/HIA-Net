# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a PyTorch implementation of **HIA-Net (Hierarchical Interactive Alignment Network)** for multimodal few-shot emotion recognition. The model processes EEG (electroencephalography) and eye-tracking data to recognize emotions in a few-shot learning setting.

Key characteristics:
- Uses prototypical networks for few-shot classification
- Implements cross-modal attention between EEG and eye-tracking features
- Applies domain adaptation via GDD (Geodesic Distance Discrepancy) loss
- Supports both SEED and SEED-Franch datasets

## Common Commands

### Training

Train on SEED dataset (default):
```bash
python train.py --cuda 0 --seed 42
```

Train on SEED-Franch dataset:
```bash
python train_franch.py --cuda 0 --seed 42
```

### Key Training Parameters

Key hyperparameters (via command line):
- `--cuda`: GPU device ID (default: 1)
- `--seed`: Random seed (default: 42)
- `--epochs`: Training epochs (default: 50)
- `--learning_rate`: Learning rate (default: 1e-4)
- `--patience`: Early stopping patience (default: 15)
- `--iterations`: Episodes per epoch (default: 20)
- `--num_support_src`: Support samples per class for source (default: 1)
- `--num_query_src`: Query samples per class for source (default: 20)
- `--num_support_tgt`: Support samples per class for target (default: 1)
- `--num_query_tgt`: Query samples per class for target (default: 20)

## Architecture Overview

### Data Flow

1. **Input**: EEG (5 bands × 9 × 9) and Eye-tracking (177-dim) features
2. **Feature Extraction**:
   - EEG: ResCBAM (ResNet + Channel/Spatial Attention) → 256-dim
   - Eye: DenseNet1D → 256-dim
3. **Cross-Modal Fusion**: MLCrossAttentionGating
   - 3-layer cross-attention with gating mechanism
   - Residual connections at each layer
   - Returns 5 intermediate outputs for multi-level GDD loss
4. **Prototypical Classification**: ProtoNet computes distances to class prototypes

### Key Components

- **ResCBAM** (`network/rescnn.py`): EEG feature extractor with CBAM attention
- **MLCrossAttentionGating** (`network/Cross_Att.py`): Cross-modal fusion with 3 attention layers
- **ProtoNet** (`network/proto_att.py`): Prototypical network for distance computation
- **GDD Loss** (`utils/gdd.py`): Domain adaptation via geodesic distance discrepancy

### Training Loss

Total loss = Proto_loss + γ × GDD_loss

Where:
- Proto_loss: Cross-entropy from prototypical network
- GDD_loss: Multi-level domain discrepancy (applied to all 5 fusion layer outputs)
- γ: Dynamic weight that increases with epochs (using sigmoid annealing)

### Data Organization

The code expects data at specific paths (configured in train.py):
- EEG: `/disk2/home/yuankang.fu/Datasets/SEED-China/02-EEG-DE-feature/eeg_used_4s`
- Eye: `/disk2/home/yuankang.fu/Datasets/SEED-China/04-Eye-tracking-feature/eye_tracking_feature`

**Important**: You must update these paths in `train.py` and `train_franch.py` to match your local data location.

### Subject-Independent Evaluation

The code implements leave-one-subject-out cross-validation:
- For each target subject, remaining subjects are used as source
- Training: source subjects + small support set from target
- Testing: query set from target subject
- Early stopping based on validation accuracy

## Important Implementation Details

1. **Session Structure**: SEED has 3 sessions, each with different subjects. The code iterates through sessions and subjects.

2. **Prototypical Batch Sampling**: Uses `PrototypicalBatchSampler` to ensure balanced class distribution in each episode.

3. **Model Checkpointing**: Early stopping saves both best and last models; final test uses the better of the two.

4. **TensorBoard**: Logs are saved to `data/tensorboard/experiment/session{X}CAMP/target{Y}/`
