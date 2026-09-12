# MLUA Official Hyperparameters Master Table

This table lists every official hyperparameter extracted directly from the verified source code of `Zzz512/MLUA`.

---

## 1. Verified Hyperparameters

| Category | Parameter | Official Source Value | Source File | Confidence |
|---|---|---|---|---|
| **Architecture** | Backbone Encoder | `resnet34` | `model/FPN.py:125` | HIGH |
| **Architecture** | Pretrained Weights | `'imagenet'` | `model/FPN.py:126` | HIGH |
| **Architecture** | Input Channels | `1` (Grayscale) | `model/FPN.py:125` | HIGH |
| **Architecture** | Output Classes | `1` (Binary caries) | `model/FPN.py:125` | HIGH |
| **Architecture** | Encoder Depth | `5` stages | `model/FPN.py:125` | HIGH |
| **Architecture** | Pyramid Channels | `256` | `model/FPN.py:126` | HIGH |
| **Architecture** | Segmentation Channels | `128` | `model/FPN.py:126` | HIGH |
| **Architecture** | Merge Policy | `"add"` (elementwise sum) | `model/FPN.py:142` | HIGH |
| **Architecture** | Decoder Dropout | `0.2` (`Dropout2d`) | `model/FPN.py:141` | HIGH |
| **Architecture** | Head Upsampling | `4x` bilinear | `model/FPN.py:148, 154` | HIGH |
| **Architecture** | Number of Aux Heads | `4` heads (P5, P4, P3, P2) | `model/FPN.py:150-155` | HIGH |
| **Optimization** | Optimizer | `AdamW` | `mlua_run.py:173` | HIGH |
| **Optimization** | Initial Learning Rate | `1e-3` ($0.001$) | `mlua_run.py:46, 215` | HIGH |
| **Optimization** | Weight Decay | Default AdamW ($0.01$) | `mlua_run.py:173` | HIGH |
| **Optimization** | LR Scheduler | Polynomial `(1 - epoch / 200) ** 0.9` | `mlua_run.py:174-175` | HIGH |
| **Optimization** | Max Epochs | `200` | `mlua_run.py:51, 256` | HIGH |
| **Optimization** | Mixed Precision | `16-bit` (`precision=16`) | `mlua_run.py:208` | HIGH |
| **Optimization** | GPU Device Count | `1` (`gpus=[0]`) | `mlua_run.py:34, 207` | HIGH |
| **Data & Sampling** | Batch Size (Total) | `8` | `mlua_run.py:220` | HIGH |
| **Data & Sampling** | Labeled Batch Size | `4` | `mlua_run.py:220` | HIGH |
| **Data & Sampling** | Unlabeled Batch Size | `4` | `mlua_run.py:220` | HIGH |
| **Data & Sampling** | Input Resolution | `384 × 384` | `clcc_run.py:43, 59` | HIGH |
| **Data & Sampling** | Random Seed | `42` | `mlua_run.py:183, 260` | HIGH |
| **Data & Sampling** | Horizontal Flip Prob | `0.5` | `clcc_run.py:55` | HIGH |
| **Data & Sampling** | Rotation Range | `45 degrees` | `clcc_run.py:57` | HIGH |
| **Data & Sampling** | Color Jitter (B / C) | `0.5` brightness, `0.5` contrast | `clcc_run.py:52` | HIGH |
| **SSL Mean Teacher** | EMA Coefficient $\theta$ | `0.99` | `mlua_run.py:46, 216` | HIGH |
| **SSL Mean Teacher** | EMA Ramp-up Formula | `min(1 - 1 / (epoch + 1), 0.99)` | `mlua_run.py:179` | HIGH |
| **SSL Mean Teacher** | EMA Frequency | Every batch (`on_train_batch_end`) | `mlua_run.py:178` | HIGH |
| **Monte Carlo SSL** | MC Passes ($T$) | `8` | `mlua_run.py:87` | HIGH |
| **Monte Carlo SSL** | Total Predictions / Sample | `40` ($8 \text{ passes} \times 5 \text{ heads}$) | `mlua_run.py:90-98` | HIGH |
| **Monte Carlo SSL** | Perturbation Distribution | Gaussian $\mathcal{N}(0, 0.01)$ | `mlua_run.py:83, 92` | HIGH |
| **Monte Carlo SSL** | Perturbation Clamping | `[-0.1, 0.1]` | `mlua_run.py:83, 92` | HIGH |
| **Monte Carlo SSL** | Dynamic Threshold Start | $0.75 \times \ln(2) \approx 0.520$ | `mlua_run.py:102` | HIGH |
| **Monte Carlo SSL** | Dynamic Threshold End | $1.00 \times \ln(2) \approx 0.693$ | `mlua_run.py:102` | HIGH |
| **Monte Carlo SSL** | Threshold Ramp-up Steps | `4480` global steps | `mlua_run.py:102` | HIGH |
| **Loss & Weights** | Supervised BCE Weight | $0.5 \times 0.25 = 0.125$ per head | `mlua_run.py:119` | HIGH |
| **Loss & Weights** | Supervised Dice Weight | $0.5 \times 0.25 = 0.125$ per head | `mlua_run.py:119` | HIGH |
| **Loss & Weights** | Max Consistency Weight | `0.1` | `util/utils.py:375` | HIGH |
| **Loss & Weights** | Consistency Ramp-up | `200` epochs | `mlua_run.py:121` | HIGH |
| **Evaluation** | Full Image Height / Width | `768 × 1536` | `evaluate/utils.py:57` | HIGH |
| **Evaluation** | Tile Size / Stride | `384 × 384` / `192 × 192` (21 patches) | `evaluate/utils.py:57, 144-146` | HIGH |
| **Evaluation** | Decision Threshold | `0.5` | `mlua_run.py:139, 140` | HIGH |
| **Evaluation** | Validation Frequency | Epochs % 10 == 0 or Epoch > 150 | `mlua_run.py:131` | HIGH |
| **Checkpoints** | Monitored Metric | `val_mean_dice` (maximize) | `mlua_run.py:201` | HIGH |
| **Checkpoints** | Top K Checkpoints | `save_top_k = 5` | `mlua_run.py:203` | HIGH |
