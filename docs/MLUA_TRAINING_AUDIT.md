# MLUA Training Pipeline & Semi-Supervised Mechanism Forensic Audit

## 1. Semi-Supervised Mean Teacher Setup

### 1.1 Model Initialization & Parameter Separation
```python
self.model_tea = Net()  # Teacher Model (ResNet-34 FPN)
self.model_stu = Net()  # Student Model (ResNet-34 FPN)
for para in self.model_tea.parameters():
    para.detach_()      # Teacher does not receive optimizer gradients
```

### 1.2 EMA Update Rule
- **Hook**: Triggered at the end of each training batch (`on_train_batch_end`).
- **Dynamic EMA Coefficient**:
  $$\alpha = \min\left(1 - \frac{1}{\text{epoch} + 1}, \theta\right), \quad \theta = 0.99$$
  *(Note: The official code computes $\alpha$ using `self.current_epoch` rather than the global step)*.
- **In-Place Weight Update**:
  ```python
  for para1, para2 in zip(self.model_tea.parameters(), self.model_stu.parameters()):
      para1.data = alpha * para1.data + (1 - alpha) * para2.data
  ```

---

## 2. Multi-Level Monte Carlo Uncertainty Estimation

### 2.1 Inputs & Forward passes
- For unlabeled batch $X_u \in \mathbb{R}^{4 \times 1 \times 384 \times 384}$:
  - Initial perturbed input: $X_{u, \text{base}} = X_u + \text{clamp}(\mathcal{N}(0, 0.01), -0.1, 0.1)$.
  - Base teacher pseudo-label: $\hat{Y}_{u, \text{base}} = \text{model\_tea}(X_{u, \text{base}})[0]$ (fused head output).

### 2.2 Monte Carlo Sampling ($T = 8$)
- Number of passes: $T = 8$.
- For each pass $i \in \{0, \dots, 7\}$:
  - Add fresh random Gaussian perturbation:
    $$\tilde{X}_u^{(i)} = X_u + \text{clamp}(\mathcal{N}(0, 0.01), -0.1, 0.1)$$
  - Teacher forward evaluation under `torch.no_grad()`:
    $$\hat{Y}_{\text{fused}}^{(i)}, \{\hat{Y}_{\text{aux}, k}^{(i)}\}_{k=1}^4 = \text{model\_tea}(\tilde{X}_u^{(i)})$$
  - All 5 predictions (1 fused + 4 pyramid auxiliary heads) are stored.
  - Across $T=8$ iterations, a total of $5 \times 8 = 40$ prediction maps of size $[4, 1, 384, 384]$ are gathered ($40 \times 4 = 160$ individual slices).

### 2.3 Uncertainty Map & Filtering Mask Computation
1. **Average Prediction**:
   $$\bar{P} = \sigma\left(\frac{1}{5T} \sum_{i=1}^{T} \left( \hat{Y}_{\text{fused}}^{(i)} + \sum_{k=1}^4 \hat{Y}_{\text{aux}, k}^{(i)} \right)\right) \in \mathbb{R}^{4 \times 1 \times 384 \times 384}$$
2. **Voxel-wise Entropy / Uncertainty**:
   ```python
   uncertainty = -2.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1, keepdim=True)
   ```
   *(Note: Computed with `dim=1` on the single-channel binary prediction tensor)*.
3. **Dynamic Ramp-up Threshold**:
   $$\beta(t) = \left(0.75 + 0.25 \times \text{sigmoid\_rampup}(t, 4480)\right) \times \ln(2)$$
   - Initial threshold (step 0): $0.75 \times \ln(2) \approx 0.520$.
   - Final threshold (step 4480): $1.00 \times \ln(2) \approx 0.693$.
4. **Confidence Mask**:
   $$M = \mathbb{I}(\text{uncertainty} < \beta(t)) \in \{0, 1\}^{4 \times 1 \times 384 \times 384}$$

---

## 3. Loss Functions & Objective Formulation

### 3.1 Supervised Deep Supervision Loss ($\mathcal{L}_{\text{sup}}$)
Computed strictly on labeled samples ($X_l, Y_l$ for indices $0..3$):
- Cross-entropy with logits: $\mathcal{L}_{\text{BCE}}(z, y) = \text{binary\_cross\_entropy\_with\_logits}(z, y)$.
- Dice loss: $\mathcal{L}_{\text{Dice}}(z, y) = \text{DiceLoss}(z, y)$ (where DiceLoss applies sigmoid internally).
- Summed across all 4 auxiliary outputs and the 1 fused head:
  $$\mathcal{L}_{\text{BCE, tot}} = \mathcal{L}_{\text{BCE}}(z_{\text{fused}}, Y_l) + \sum_{k=1}^4 \mathcal{L}_{\text{BCE}}(z_{\text{aux}, k}, Y_l)$$
  $$\mathcal{L}_{\text{Dice, tot}} = \mathcal{L}_{\text{Dice}}(z_{\text{fused}}, Y_l) + \sum_{k=1}^4 \mathcal{L}_{\text{Dice}}(z_{\text{aux}, k}, Y_l)$$
- **Exact Code Weighting**:
  $$\mathcal{L}_{\text{seg}} = 0.5 \times \left(\frac{\mathcal{L}_{\text{BCE, tot}}}{4} + \frac{\mathcal{L}_{\text{Dice, tot}}}{4}\right)$$
  *(Note: Divided by 4 even though 5 loss terms are summed)*.

### 3.2 Consistency Loss ($\mathcal{L}_{\text{cons}}$)
Between student unlabeled fused prediction $z_{u, \text{stu}}$ and base teacher prediction $\hat{Y}_{u, \text{tea}}$:
$$\mathcal{D}_{\text{MSE}} = (\sigma(z_{u, \text{stu}}) - \sigma(\hat{Y}_{u, \text{tea}}))^2$$
$$\mathcal{L}_{\text{cons}} = \frac{\sum (M \odot \mathcal{D}_{\text{MSE}})}{2 \sum M + 10^{-16}}$$

### 3.3 Consistency Weight Ramp-up ($\lambda(e)$)
$$\lambda(e) = 0.1 \times \exp\left(-5 \left(1 - \min\left(1, \frac{e}{200}\right)\right)^2\right)$$
- Epoch 0: $\lambda(0) = 0.1 \times e^{-5} \approx 0.00067$.
- Epoch 200: $\lambda(200) = 0.10$.

### 3.4 Total Objective
$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{seg}} + \lambda(e) \cdot \mathcal{L}_{\text{cons}}$$

---

## 4. Optimization & Schedule

- **Optimizer**: `AdamW(self.parameters(), lr=1e-3)`
- **Weight Decay**: Default AdamW ($0.01$)
- **Learning Rate Scheduler**:
  $$\text{lr}(e) = \text{lr}_0 \times \left(1 - \frac{e}{200}\right)^{0.9}$$
  - Stepped once per epoch via `LambdaLR`.
- **Total Training Epochs**: 200.
- **Precision**: 16-bit mixed precision (`precision=16` in PyTorch Lightning Trainer).
- **Gradient Clipping**: None specified in Trainer args.
- **Gradient Accumulation**: None (`accumulate_grad_batches=1`).
