# EXP-MLUA-003: Teacher BatchNorm Buffer Synchronization
## Official Controlled Remediation Experiment

---

### 1. Experiment ID
`EXP-MLUA-003`

### 2. Parent Baseline
`EXP-MLUA-002` (Frozen at Epoch 9, baseline preserved byte-identical)

### 3. Intended Intervention
**Teacher BatchNorm Buffer EMA Synchronization**:
In the Mean Teacher update loop after `optimizer.step()`, synchronize `model_tea.buffers()` alongside `model_tea.parameters()`:
```python
# Parameter EMA
alpha = min(1.0 - 1.0 / (epoch + 1), theta)
for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):
    p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0 - alpha)

# Buffer EMA Synchronization (Key Intervention)
for b_tea, b_stu in zip(model_tea.buffers(), model_stu.buffers()):
    if b_tea.dtype.is_floating_point:
        b_tea.data.mul_(alpha).add_(b_stu.data, alpha=1.0 - alpha)
    else:
        b_tea.data.copy_(b_stu.data)
```

### 4. Scientific Hypothesis
- **H0**: Synchronizing Teacher buffers will not materially improve numerical stability.
- **H1 (Primary)**: Correct synchronization of Teacher BatchNorm buffers during EMA will prevent the exponential activation compounding in the Teacher encoder, eliminate the FP32 GroupNorm variance overflow observed in EXP-MLUA-002, and allow stable training to proceed through Epoch 10 (Batch 68 / Step 1257) and beyond while preserving the entire MLUA semi-supervised methodology.

### 5. Controlled Unchanged Variables (100% Identical to EXP-MLUA-002)
- **Dataset**: DC1000 oral ulcer dataset (2,389 training patches, single-channel grayscale, $384 \times 384$).
- **Partition**: 530 labeled / 1,859 unlabeled patches (20% semi-supervised DICE530 setting).
- **Random Seed**: `42` across Python, NumPy, and PyTorch CPU/CUDA.
- **Architecture**: `Net` (Standalone ResNet-34 encoder + FPN decoder, 4 auxiliary heads + 1 fused head, GroupNorm segmentation blocks).
- **Precision**: 100% Native FP32 (zero AMP / autocast).
- **Loss Functions**: Supervised BCE + Dice Loss ($0.5 \times (\text{BCE} + \text{Dice})$ across all 4 auxiliary + fused heads), Unsupervised Consistency Loss via MSE with dynamic uncertainty masking and sigmoid rampup.
- **SSL MC Sampling**: $T=8$ Monte Carlo perturbations ($\sigma=0.01$, clamp $= 0.1$), uncertainty threshold $\in [0.75\ln 2, 1.00\ln 2]$ ramped over 4,480 steps.
- **Optimizer**: `AdamW` ($\text{lr} = 0.001$, $\text{betas} = [0.9, 0.999]$, $\text{weight\_decay} = 0.01$).
- **Learning Rate Scheduler**: `LambdaLR` polynomial decay with power $0.9$ over 200 epochs.
- **Batch Composition**: Batch size 8 (4 labeled + 4 unlabeled via `TwoStreamBatchSampler`).
- **Validation Protocol**: 50 labeled subset, patch-level evaluation at threshold $\tau = 0.50$.
- **Sealed Test Set**: 100 test patient cases sealed and completely untouched.

### 6. Pre-Flight Status
- **Pre-Flight Validation**: `PASSED` (0 unintended configuration differences).
- **In-Memory Unit Test**: `PASSED` (Teacher BN buffers verified tracking Student running statistics).
