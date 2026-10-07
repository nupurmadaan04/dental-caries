# Dataset Analysis for Mentor Review

## 1. Dataset Used
We use the **DC1000** benchmark dataset (Dental Caries 1,000 Panoramic Radiographs), acquired from the Department of Stomatology, Zhejiang Provincial People's Hospital. It represents full-mouth orthopantomogram (OPG) radiographs with pixel-level caries annotations verified by five licensed dental practitioners.

---

## 2. Dataset Size
- **Total Panoramic Radiographs:** 1,000 full-mouth OPGs.
- **Annotated Cases (Detailed):** 593 cases containing 4,520 verified lesions (average 7.6 lesions per image).
- **Unlabeled / Rough Cases:** 407 cases used as unannotated data for semi-supervised consistency training.
- **Repository Archive (`DC1000_dataset.zip`):** 900 training OPGs + 100 sealed test OPGs = 1,000 images total (~1.02 GB uncompressed).

---

## 3. Image and Mask Structure
- **Input Images:** Grayscale panoramic radiographs standardized to $768 \times 1536$ pixels (preserving natural 1:2 dental arch geometry).
- **Target Masks:** Single-channel binary masks ($0 = \text{sound tissue / background}$, $1 = \text{caries lesion}$).
- **Training Patch Geometry:** Decomposed into $384 \times 384$ pixel patches using a sliding window with a $192\text{-pixel}$ stride (exactly 21 patches per panoramic image).

---

## 4. Train / Validation / Test Split
- **Training Pool:** 2,389 total patches extracted from the training cohort.
  - **Labeled Training Subset (20%):** 530 patches with full binary ground truth.
  - **Unlabeled Training Subset (80%):** 1,859 patches trained via consistency regularization.
- **Validation Set:** 50 patches held out to monitor epoch-to-epoch convergence and select best checkpoints.
- **Independent Sealed Test Set:** Exactly 100 full-mouth panoramic radiographs ($768 \times 1536$), isolated from training and evaluated once at $\tau = 0.50$ without test-time tuning.

---

## 5. Caries Pixel Distribution
- **On Full Panoramic Images ($768 \times 1536$):** Ground-truth caries occupy an average of only **$\approx 0.15\%$ (1.5‰)** of total pixels ($\approx 6,335$ out of $1,179,648$ pixels). Over $99.85\%$ of pixels belong to sound background.
- **On $384 \times 384$ Training Patches:** Positive patch sampling increases average lesion density to **$11.79\%$** ($\approx 80\times$ increase), allowing stable gradient backpropagation.
- **Lesion Scale Variability (Base Paper Anchors):**
  - Shallow enamel lesions: $\approx 160\text{ pixels}$ ($< 300\text{ px}$).
  - Middle dentin lesions: $\approx 474\text{ pixels}$ ($300 - 1000\text{ px}$).
  - Deep pulp-approaching lesions: $\approx 1,952\text{ pixels}$ ($> 1000\text{ px}$).

---

## 6. Main Dataset Problems

### A. Severe Foreground / Background Imbalance
With positive pixels at only $0.15\%$, standard cross-entropy loss causes the network to trivially predict all background ($0\text{ positive pixels}$), achieving $99.85\%$ accuracy while learning nothing.

### B. Minute Lesion Size
Incipient lesions can be as small as $20 - 50$ pixels. Standard heavy downsampling (e.g., standard pooling without skip connections) completely erases these micro-signals before they reach the decoder.

### C. Cervical Burnout Artifacts
The physiological narrowing at the tooth neck creates an anatomical radiolucency (dark shadow) that resembles interproximal decay, confusing models relying only on local contrast.

### D. Ambiguous Lesion Boundaries
Demineralization is a progressive chemical process without sharp edges. The transition between sound dentin, demineralized dentin, and caries is gradual, leading to inter-annotator boundary divergence.

### E. Restorations and Radiopaque Implants
Metallic fillings produce bright streak artifacts, while composite resin fillings appear dark (radiolucent), mimicking active caries on 2D radiographs.

---

## 7. Why This Dataset Requires SSL and Uncertainty Handling

1. **Why Semi-Supervised Learning (SSL)?**
   Only 530 patches (20%) have verified labels. Training purely supervised on 530 patches causes severe overfitting (EXP001 achieved only 54.21% validation Dice). SSL allows the network to learn rich anatomical structures from the 1,859 unannotated patches (80%) without requiring expensive manual dental annotations.

2. **Why Uncertainty Handling?**
   Because cervical burnout and diffuse boundaries generate noisy pseudo-labels, standard semi-supervised consistency would force the student network to memorize false positives. Monte Carlo uncertainty gating estimates prediction variance and suppresses loss updates on ambiguous pixels, ensuring the model only learns from confident regions.
