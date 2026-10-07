# Why EXP-MLUA-003? (Design & Architectural Rationale)

This document provides concise, 1-3 sentence answers to the core questions our mentor will ask regarding why our final model and experimental configuration were chosen.

---

### Starting Problem: What Was Wrong with the Previous Approach?
Fully supervised models hit an empirical plateau below 48.4% Dice because only 530 annotated patches were available, leading to severe overfitting. When we attempted semi-supervised learning in EXP-002, the model collapsed with NaN loss at Step 1257 due to an EMA buffer bug.

### Research Motivation: Why MLUA?
The MLUA (Multi-Level Uncertainty-Aware) framework by Wang et al. (2023) allows the model to leverage 1,859 unannotated clinical patches (80% of the cohort) while filtering out noisy pseudo-labels using uncertainty gating. This directly overcomes the dual challenges of data scarcity and ambiguous lesion boundaries.

### Implementation Decision: Why ResNet-34 + FPN?
ResNet-34 provides ImageNet-pretrained feature transfer without the overfitting risks of heavier backbones like ResNet-50 on our limited dataset. The Feature Pyramid Network (FPN) lateral connections fuse shallow high-resolution spatial boundaries with deep semantic context, which is necessary to detect tiny enamel demineralizations.

### Why Teacher / Student?
A Teacher network updated by Exponential Moving Average (EMA) produces temporally smoothed, highly stable target predictions for unlabeled patches. This consistency regularization forces the Student to learn representations that are invariant to noise and scanner differences without needing manual labels.

### Why Uncertainty?
Demineralization boundaries are diffuse, and panoramic radiographs contain confounding shadows such as cervical burnout that resemble caries. Uncertainty estimation identifies pixels where the model is hesitant and masks them out, preventing false-positive noise from polluting the consistency loss.

### Why Monte Carlo (MC) Sampling?
Running $T=8$ stochastic forward passes under active dropout ($p=0.20$) approximates Bayesian inference over network weights. The pixel-wise variance across these passes directly quantifies epistemic (model) uncertainty without requiring expensive Bayesian neural network architectures.

### Why Deep Supervision?
Small caries lesions can be as small as 20 pixels and their gradient signals vanish across deep backbones. Attaching auxiliary prediction heads at 4 intermediate decoder scales ($1/8, 1/4, 1/2, 1/1$) injects direct gradient feedback into early layers and enforces multi-scale representation learning.

### Why Patch-Based Processing?
On full $768 \times 1536$ panoramic images, caries occupy only $0.15\%$ (1.5‰) of pixels, causing standard loss gradients to collapse to predicting all background. Decomposing images into $384 \times 384$ patches amplifies positive lesion density by $\approx 80\times$ (to $11.79\%$), restoring healthy gradient flow.

### Why 21 Patches?
Extracting $384 \times 384$ patches with a $192\text{-pixel}$ stride (50% horizontal and vertical overlap) across a $768 \times 1536$ canvas yields exactly $3 \text{ rows} \times 7 \text{ columns} = 21\text{ patches}$. This 50% overlap guarantees that no dental interproximal contact is cut by patch borders and enables smooth 2D Gaussian reconstruction during full-mouth inference.

### Why Synchronized Teacher BatchNorm Buffers?
Standard PyTorch EMA only updates trainable parameters, leaving BatchNorm `running_mean` and `running_var` static. Synchronizing floating-point buffers alongside parameters prevents statistical distribution drift between Student and Teacher, completely eliminating the $10^{18}$ activation explosion that ruined EXP-002.

### Why Did We Freeze the Experimental Setup?
Once EXP-003 achieved stable convergence, we strictly froze all random seeds (Seed 42), split allocations, and hyperparameters to ensure scientific reproducibility and prevent data leakage onto the independent sealed test set.
