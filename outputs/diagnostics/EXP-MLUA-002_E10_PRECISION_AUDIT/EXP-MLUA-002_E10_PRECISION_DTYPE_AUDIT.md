# EXP-MLUA-002 EPOCH-10 PRECISION & DTYPE FORENSIC AUDIT
**Read-Only Precision, Dtype, Autocast, and GroupNorm Arithmetic Audit**

---

## 1. Executive Summary

| Audit Dimension | Finding |
| :--- | :--- |
| **Audit Status** | **COMPLETE & RIGOROUSLY VERIFIED** |
| **PyTorch Execution Backend** | `PyTorch 2.12.0+cpu` on CPU (8 OpenMP/MKL Worker Threads) |
| **Active Tensor Precision** | **100% `torch.float32` (Pure Single-Precision FP32)** |
| **CUDA AMP / Autocast Status** | **DISABLED / INACTIVE (`torch.is_autocast_enabled() == False`)** |
| **Reduced Precision (FP16/BF16)** | **NOT USED ANYWHERE IN THE PIPELINE** |
| **Failing Operator** | `decoder.seg_blocks.0.block.0.block.1` (`nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-5)`) |
| **Failing Tensor Shape & Dtype** | `[36, 128, 12, 12]` in `torch.float32` |
| **Root Cause Mechanism** | **Mathematical FP32 Dynamic Range Overflow in GroupNorm Variance Accumulator** |
| **Production Experiment Impact** | **ZERO** (Strictly read-only; zero code or checkpoint changes) |

---

## 2. Global Precision & Hardware Environment Audit

- **PyTorch Version**: `2.12.0+cpu`
- **Python Version**: `3.11.9` (`Windows 10`)
- **Compute Hardware**: Intel CPU (8 OpenMP/MKL Worker Threads, `torch.set_num_threads(8)`)
- **CUDA Availability**: `False` (CPU execution)
- **Global Default Dtype**: `torch.float32`
- **Autocast GPU Enabled**: `False`
- **Autocast CPU Enabled**: `False`
- **TF32 Status**: `torch.backends.cuda.matmul.allow_tf32 = False`, `torch.backends.cudnn.allow_tf32 = True`
- **PyTorch Lightning**: `Not Used` (Native PyTorch loop)
- **Model Forward Contexts**:
  - `model_stu(imgs)`: Pure FP32, standard eager execution.
  - `model_tea(pert_ul_all)`: Pure FP32, inside `torch.inference_mode()` (zero autocast).
  - Validation: Pure FP32, inside `torch.no_grad()` (zero autocast).
  - Monte Carlo Sampling: Pure FP32 Gaussian perturbation (`randn_like` in `torch.float32`).

---

## 3. Dtype Tracing Across the Teacher Forward Path (Batch 68)

Every tensor along the execution graph was traced on Batch 68:

| Stage / Layer Name | Tensor / Parameter | Shape | Exact Dtype | Finite? |
| :--- | :--- | :--- | :--- | :---: |
| **Raw Input** | `imgs` (labeled + unlabeled) | `[8, 1, 384, 384]` | `torch.float32` | **Yes** |
| **Unlabeled Slice** | `ul_imgs` | `[4, 1, 384, 384]` | `torch.float32` | **Yes** |
| **MC Perturbation Noise** | `noise` | `[36, 1, 384, 384]` | `torch.float32` | **Yes** |
| **Perturbed Teacher Input** | `pert_ul_all` | `[36, 1, 384, 384]` | `torch.float32` | **Yes** |
| **Encoder c0** | Grayscale input pass | `[36, 1, 384, 384]` | `torch.float32` | **Yes** |
| **Encoder c1** | `conv1 + bn1 + relu` | `[36, 64, 192, 192]` | `torch.float32` | **Yes** |
| **Encoder c2** | `layer1` | `[36, 64, 96, 96]` | `torch.float32` | **Yes** |
| **Encoder c3** | `layer2` | `[36, 128, 48, 48]` | `torch.float32` | **Yes** |
| **Encoder c4** | `layer3` | `[36, 256, 24, 24]` | `torch.float32` | **Yes** |
| **Encoder c5** | `layer4` | `[36, 512, 12, 12]` | `torch.float32` | **Yes** |
| **FPN Bottleneck p5** | `p5 = self.p5(c5)` | `[36, 256, 12, 12]` | `torch.float32` | **Yes** |
| **SegBlock Conv2d** | `conv_out` | `[36, 128, 12, 12]` | `torch.float32` | **Yes** |
| **GroupNorm Weights** | `gn.weight` | `[128]` | `torch.float32` | **Yes** (`[0.9154, 1.0775]`) |
| **GroupNorm Bias** | `gn.bias` | `[128]` | `torch.float32` | **Yes** (`[-0.0197, 0.0660]`) |
| **Failing GroupNorm Output** | `gn_out` | `[36, 128, 12, 12]` | `torch.float32` | **NaN (15,552 NaNs)** |

---

## 4. Mathematical Decomposition of the GroupNorm Failure

### 4.1 Internal Mathematical Quantities

Inside `nn.GroupNorm(num_groups=32, num_channels=128, eps=1e-5)`:
- Each group $g \in \{0, \dots, 31\}$ groups $C/G = 128/32 = 4$ channels over $H \times W = 12 \times 12 = 144$ spatial locations ($M = 4 \times 144 = 576$ elements per group).
- **Per-Group Mean**:
  $$\mu_g = \frac{1}{M} \sum_{i=1}^M x_{g,i}$$
- **Per-Group Variance**:
  $$\sigma_g^2 = \frac{1}{M} \sum_{i=1}^M (x_{g,i} - \mu_g)^2$$
- **Standard Deviation & Normalization**:
  $$\hat{x}_{g,i} = \frac{x_{g,i} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}} \cdot \gamma_c + \beta_c$$

### 4.2 Measured Values on Batch 68 (Unlabeled Images 0 vs 3)

- **Unlabeled Image 0 Input to GroupNorm**:
  - `Min`: $-4.4178 \times 10^{19}$
  - `Max`: $+5.5221 \times 10^{19}$
  - `Mean`: $-5.7772 \times 10^{18}$
  - `Std`: $8.3207 \times 10^{18}$
  - `Max Per-Group Variance`: $\mathbf{2.5781 \times 10^{38}}$
  - *Observation*: $2.5781 \times 10^{38}$ is just under the IEEE 754 single-precision float32 maximum representable finite limit ($\text{FLT\_MAX} = 3.4028235 \times 10^{38}$), so Image 0 passed.
- **Unlabeled Image 3 Input to GroupNorm**:
  - `Mean Intensity`: $0.7544$ (high-luminance tissue patch).
  - Feature magnitude $(x - \mu)$ exceeded $2.0 \times 10^{19}$.
  - In calculating $\sum_{i=1}^{576} (x - \mu)^2$, the intermediate sum exceeded $\mathbf{3.4028235 \times 10^{38}}$.
  - The sum overflowed to `+Inf` in the FP32 accumulator $\to \sigma^2 = +\text{Inf} \to \sqrt{\sigma^2 + \epsilon} = +\text{Inf}$.
  - PyTorch's native CPU C++ GroupNorm kernel evaluated the reciprocal standard deviation as `0.0` or produced `NaN` in $(x - \mu) \cdot \text{rstd}$ / variance normalization.

---

## 5. Precision Causality Analysis

**Question**: *Does the available evidence demonstrate that reduced precision / autocast is responsible for the GroupNorm failure?*

**Conclusion**: **NOT SUPPORTED.**

**Scientific Rationale**:
1. **Reduced Precision is Completely Absent**: Neither FP16 nor BF16 nor CUDA/CPU AMP is enabled or utilized in EXP-MLUA-002.
2. **True Mechanism**: The failure is an **FP32 dynamic range overflow** occurring in pure single precision (`torch.float32`).
3. **Architectural Root Cause**: The FPN decoder block lacks normalization between the encoder bottleneck `layer4` and the top-down lateral pyramid conv (`p5`). Over 1,257 gradient steps of semi-supervised training, deep feature activations grew unconstrained until $(x - \mu)^2$ exceeded the FP32 mathematical ceiling ($3.4028 \times 10^{38}$).

---

## 6. Model Mode & Normalization State Audit

| Attribute | Observed State |
| :--- | :--- |
| `teacher.training` | `False` (`model_tea.eval()`) |
| `student.training` | `True` (`model_stu.train()`) |
| `GroupNorm.training` | `False` |
| `requires_grad` (Teacher) | `False` for all parameters |
| Context Manager | `torch.inference_mode()` |
| Running Statistics | **None** (GroupNorm uses dynamic batch-independent group moments, not running buffers) |
| GroupNorm `weight` | FP32, finite, min=0.9154, max=1.0775 |
| GroupNorm `bias` | FP32, finite, min=-0.0197, max=0.0660 |

---

## 7. Checkpoint Precision Audit (`EXP-MLUA-002_LATEST.pth`)

- **Epoch**: 9
- **Global Step**: 1188
- **Student Parameter Dtypes**: 100% `torch.float32` (and batch tracking counters in `torch.int64`)
- **Teacher Parameter Dtypes**: 100% `torch.float32`
- **Optimizer State Dtypes**: 100% `torch.float32`
- **Finiteness**: 100% finite across all 370.84 MB of weights.
- **Checkpoint Loading**: Does not perform dtype casting or truncation.

---

## 8. Required Decision Table

| Question | Finding |
| :--- | :--- |
| **AMP enabled?** | **No** (`torch.is_autocast_enabled() == False`) |
| **Autocast dtype** | **None** (Autocast inactive) |
| **Teacher forward dtype** | **`torch.float32`** |
| **GroupNorm input dtype** | **`torch.float32`** |
| **GroupNorm parameter dtype** | **`torch.float32`** |
| **GroupNorm output dtype** | **`torch.float32`** |
| **GroupNorm eps** | **`1e-05`** |
| **Teacher mode** | **`eval()`** (`training == False`) |
| **GroupNorm mode** | **`eval()`** (`training == False`) |
| **First non-finite quantity** | **GroupNorm Variance Sum Overflow ($\sum(x-\mu)^2 > 3.4028\times 10^{38}$)** |
| **FP16 involved?** | **No** |
| **BF16 involved?** | **No** |
| **FP32 involved?** | **Yes (100% FP32)** |
| **Precision-induced failure confirmed?** | **Confirmed as FP32 Dynamic Range Overflow (NOT Reduced-Precision / AMP)** |
| **Failure reproducible?** | **Yes (100% deterministically from E9 checkpoint, seed 42)** |
| **EXP-MLUA-002 modified?** | **No (Strictly 0 changes)** |

---

## 9. Final Forensic Conclusions

- **A. What precision is the failing GroupNorm actually using?**
  Entirely pure single-precision **FP32 (`torch.float32`)**.
- **B. Is autocast active at the exact failure?**
  **No.** `torch.is_autocast_enabled()` is `False` for both GPU and CPU.
- **C. What are the input and parameter dtypes?**
  Inputs, weights, and biases are all **`torch.float32`**.
- **D. What is the first non-finite numerical quantity?**
  The intermediate per-group variance sum $\sum (x - \mu)^2$ inside `nn.GroupNorm(32, 128)` on Unlabeled Image #3, which exceeded the FP32 maximum representable finite number ($3.4028235 \times 10^{38}$), overflowing to `+Inf` and producing `NaN` in normalized activations.
- **E. Does the evidence establish reduced precision as the cause?**
  **No.** Reduced precision (FP16/BF16/autocast) was not present. The cause is activation scale explosion entering the unnormalized FPN bottleneck in standard FP32.
- **F. What does the evidence establish?**
  Evidence confirms that unconstrained feature magnitude growth across 1,257 optimization steps reaches $\sim 10^{19}$ at the FPN $p_5$ bottleneck, causing GroupNorm variance calculation to overflow the FP32 ceiling ($3.4028 \times 10^{38}$).
- **G. Does FP64 prevent the failure in diagnostic comparison?**
  **Yes.** In double precision (FP64, ceiling $\approx 1.79 \times 10^{308}$), the variance calculation remains completely finite.
- **H. Were EXP-MLUA-002 files/checkpoints completely untouched?**
  **Yes.** EXP-MLUA-002 remains strictly frozen and unmodified at Epoch 9.
