# FORENSIC AUDIT: MLUA DENTAL CARIES INFERENCE & FRONTEND PIPELINE

**Audit Date**: September 9, 2026  
**Auditor**: Forensic ML & Systems Audit Agent  
**Target Repository**: `MLUA` / Dental Caries AI Clinical Decision Support System  
**Audit Scope**: End-to-end trace from UI upload to model execution, checkpoint verification, and scientific validity analysis.  
**Constraint Enforced**: Zero model training, zero modifications to model/checkpoints/configs, zero access or evaluation on the sealed 100-case benchmark set.

---

## 1. Executive Summary & Verdict

| Audit Dimension | Forensic Finding | Evidence Reference |
|---|---|---|
| **Current Analysis Status** | **D) Client-Side Canvas Simulation (Mock / Procedural)** + **C) Disconnected Backend** + **B) Incomplete Epoch-1 Model in Repo** | `src/services/api.ts:98-165`, `src/utils/maskGenerator.ts:7-173` |
| **Is Model Trained?** | **NO** (Stopped after Epoch 1 of 200; 66 batches total) | `checkpoints/EXP-MLUA-001_BEST.pth` (`epoch=1`, `val_dice=0.0`) |
| **Is Backend Active?** | **NO** (No FastAPI/PyTorch inference server listening on port 8000) | Port 8000 connection fails immediately; caught in `try...catch` |
| **Is Current UI using Checkpoint?** | **NO** (Frontend runs purely standalone via HTML5 Canvas generation) | `src/utils/maskGenerator.ts` |
| **Sealed Test Set Accessed?** | **NO** (100-case sealed benchmark strictly unaccessed) | `dataset/test/` unopened and uncomputed |
| **Production/Clinical Validity?** | **NO** (Predictions are simulated client-side demonstrations and cannot be used for clinical diagnosis) | Synthetic procedural overlays on HTML5 Canvas |

---

## 2. End-to-End Pipeline Execution Trace

The table below traces every component from radiograph upload to clinical result rendering:

| Pipeline Stage | Responsible File | Function / Class | Actual Operational Status |
|---|---|---|---|
| **1. File Upload** | `src/pages/NewAnalysisPage.tsx` | `handleFileSelect` / `handleDrop` | **REAL** (Reads genuine user image file via browser `FileReader`) |
| **2. Analysis Trigger** | `src/pages/NewAnalysisPage.tsx` | `handleStartAnalysis` | **REAL** (Dispatches file and sensitivity threshold) |
| **3. API Service Request** | `src/services/api.ts` | `apiService.analyzeRadiograph` | **ATTEMPTED REAL &rarr; FALLBACK** (Calls `POST http://localhost:8000/api/analyze`) |
| **4. Network Layer** | `src/services/api.ts` | `axios.post` | **FAILED (OFFLINE)** (Connection refused on port 8000; caught by `catch` block) |
| **5. Inference Server** | *Non-existent / Offline* | *FastAPI / Uvicorn endpoint* | **MOCK / OFFLINE** (No live Python inference server running) |
| **6. Checkpoint Loading** | `checkpoints/EXP-MLUA-001_BEST.pth` | `torch.load` | **DISCONNECTED** (Weights exist on disk at Epoch 1, but are NOT loaded by frontend) |
| **7. PyTorch Model Forward** | `model/FPN.py`, `src/mlua/models/` | `Net.forward` / `model.eval()` | **DISCONNECTED** (Not executed during frontend analysis) |
| **8. Mask Generation** | `src/utils/maskGenerator.ts` | `generateAnatomicalCariesAnalysis` | **PROCEDURAL CANVAS SIMULATION** (Generates binary mask and cyan glow via HTML5 2D Canvas) |
| **9. Lesion Extraction** | `src/utils/maskGenerator.ts` | `drawOrganicLesion` & `findings.push` | **PROCEDURAL HEURISTIC** (Synthesizes FDI tooth numbers, bounding boxes, and stages) |
| **10. Persistence Store** | `src/utils/storageDb.ts` | `idbSaveAnalysis` / IndexedDB | **REAL** (Persists base64 canvas masks faithfully in browser IndexedDB) |
| **11. Result Rendering** | `src/pages/AnalysisResultPage.tsx` | `AnalysisResultPage` component | **REAL UI / PROCEDURAL DATA** (Renders the 3 image layers, metrics, and clinical forms) |

---

## 3. Frontend Fallback & Mock Data Forensic Inspection

Inspection of `frontend/src/services/api.ts` revealed the exact mechanism governing the analysis output:

```typescript
// frontend/src/services/api.ts (lines 97-115)
async analyzeRadiograph(file: File, threshold: number = 0.5): Promise<AnalysisResult> {
  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold.toString());

    // 1. Attempt to communicate with live PyTorch FastAPI backend
    const response = await client.post('/api/analyze', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  } catch {
    // 2. BACKEND OFFLINE FALLBACK:
    // Simulates realistic clinical latency (2200ms) and executes client-side canvas segmentation
    await new Promise((res) => setTimeout(res, 2200));

    const generated = await generateAnatomicalCariesAnalysis(file, threshold);
    const newId = `CA-${Date.now().toString().slice(-6)}`;
    ...
```

### Forensic Observations:
1. **Zero Active Python Server**: There is no live Python FastAPI server running in the background. The axios HTTP POST request unconditionally fails and triggers the `catch` block.
2. **Client-Side Generation**: The masks and bounding boxes are generated purely in JavaScript via HTML5 2D Canvas methods (`ctx.ellipse`, `ctx.strokeRect`, `drawOrganicLesion`) inside `src/utils/maskGenerator.ts`.
3. **Hardcoded Scientific Metrics**: The model evaluation metrics displayed on Tab 3 / Technical Verification (`Dice = 84.2%`, `IoU = 72.8%`, `Precision = 86.5%`, `Recall = 82.4%`, `F1 = 84.4%`) are static constant fields from published literature benchmarks (`mockData.ts`), not metrics calculated live from the uploaded image.

---

## 4. Checkpoint Forensic Examination

Direct binary inspection of the checkpoints located in `checkpoints/` and `outputs/experiments/EXP-MLUA-001/checkpoints/` using PyTorch checkpoint metadata extraction:

### Checkpoint 1: `EXP-MLUA-001_BEST.pth`
* **Path**: `c:\Users\devin\MLUA\checkpoints\EXP-MLUA-001_BEST.pth`
* **File Size**: `370,837,491 bytes` (~353.66 MB)
* **Architecture**: Official MLUA `Net` (ResNet-34 Feature Pyramid Network with 4 auxiliary deep supervision heads)
* **Parameter Count**: ~23.8 Million parameters (Weights for `model_stu` + `model_tea` + optimizer states)
* **Training Epoch**: **`1`** (Epoch 1 of 200)
* **Global Optimization Steps**: **`66`** updates (265 labeled patches / 4 labeled per batch)
* **Optimizer State**: AdamW (`lr = 0.0009955`, `weight_decay = 0.01`)
* **Training Loss**: `0.66914` (Supervised: `0.66914`, Consistency: `0.00121`)
* **Validation Loss**: `1.04736`
* **Validation Dice**: **`0.0000`**
* **Validation IoU**: **`0.0000`**
* **Validation Precision / Recall**: **`0.0000 / 0.0000`**
* **Validation Specificity**: **`1.0000`** (Predicting 100% background / 0% foreground)
* **Predicted Foreground Prevalence**: **`0.0000%`** (Ground truth prevalence: `1.04%`)

### Checkpoint 2: `EXP-MLUA-001_LATEST.pth`
* **Path**: `c:\Users\devin\MLUA\checkpoints\EXP-MLUA-001_LATEST.pth`
* **File Size**: `370,839,661 bytes` (~353.66 MB)
* **Training Epoch**: **`1`** (Identical state from Epoch 1 termination)

---

## 5. Training Status & Convergence Audit

| Training Metric | Configuration / Planned | Actual Completed in Checkpoint | Convergence Established? |
|---|---|---|---|
| **Total Epochs** | 200 epochs | **1 epoch** | **NO** (0.5% of planned training) |
| **Total Optimizer Steps** | 13,200 steps | **66 steps** | **NO** |
| **Validation Dice ($N=50$ patches)** | Target > 80% | **0.00%** | **NO** |
| **Uncertainty Rampup ($\lambda_{\text{cons}}$)** | 200 epochs ($0 \to 0.1$) | Epoch 1: $\lambda = 0.0005$ | **NO** |
| **Pseudo-label Threshold** | Rampup $0.75 \to 1.00$ | Epoch 1: $0.750$ | **NO** |
| **EMA Teacher Weights** | $\theta = 0.99$ over 200 epochs | 66 update iterations | **NO** |

### Scientific Finding on Model State:
At Epoch 1, the neural network weights are dominated by initial ImageNet backbone features and randomly initialized auxiliary heads. The network has not yet learned to segment caries, which is proven by its validation foreground prevalence of `0.0%` (Dice = `0.000`).

---

## 6. Preprocessing & Radiograph Resolution Analysis

Review of training preprocessing vs. clinical panoramic deployment:

1. **Training Patching (`configs/mlua_default.yaml`)**:
   * Training images are processed as **`384x384` pixel patches** sampled from full panoramic radiographs.
   * Full OPG dimensions in DC1000 are **`1536x768` pixels** (divided into 21 overlapping 384x384 patches with stride 192 for inference).
2. **Inference Direct Resizing Issue**:
   * If a full panoramic radiograph ($1536 \times 768$ or $3000 \times 1500$) were directly resized down into a single $384 \times 384$ square input without patch extraction, small micro-lesions (often only 5–15 pixels wide) would be obliterated by downsampling interpolation.
   * The official MLUA evaluation pipeline (`evaluate/utils.py`) uses a patch-and-recompose sliding window algorithm (`recompone_overlap`).

---

## 7. Categorized Breakdown of Current Frontend Output

| Category | Elements |
|---|---|
| **REAL USER DATA** | - Uploaded image binary & dimensions<br>- Selected detection sensitivity threshold slider ($0.10 - 0.90$)<br>- Radiograph file metadata (filename, upload date/time, file size) |
| **CLIENT-SIDE PROCEDURAL SIMULATION** | - Candidate lesion overlay (Cyan glowing boundary contour)<br>- Binary segmentation mask (White organic shape on solid black background)<br>- FDI tooth number attribution (`Tooth 46`, `Tooth 16`, `Tooth 36`)<br>- Candidate site bounding boxes (`bbox: [x, y, w, h]`)<br>- Affected area percentage & pixel count calculation<br>- Dynamic Stage classification (`Stage 0`, `Stage 1`, `Stage 2`, `Stage 3`) |
| **HARDCODED LITERATURE BENCHMARKS** | - Academic evaluation metrics card & tables (`Dice: 84.2%`, `IoU: 72.8%`, `Precision: 86.5%`, `Recall: 82.4%`, `F1: 84.4%`)<br>- Evaluation cohort reference (`DC1000 Panoramic Cohort, N=100`) |
| **DISCONNECTED / NOT EXECUTED** | - PyTorch model forward pass (`model_stu(img)`)<br>- Monte Carlo $T=8$ uncertainty variance map (`\sigma^2(x)`)<br>- Deep supervision multi-scale aux head logits ($H_1, H_2, H_3, H_4$) |

---

## 8. Final Forensic Diagnosis & Answer

### Question:
> *"Is the current analysis primarily because the MLUA model has not completed training, or because the application is not actually using the model correctly?"*

### Forensic Answer:
**BOTH, with the primary immediate factor being that the application is running in standalone client-side simulation mode without an active PyTorch backend.**

1. **Disconnected Backend**: The frontend application does not currently have an active Python backend server connected. When you upload a radiograph, the frontend executes a client-side procedural canvas algorithm (`generateAnatomicalCariesAnalysis`) to demonstrate the clinical workflow, UI layers, and physician review tools.
2. **Incomplete Checkpoint on Disk**: Even if a FastAPI backend were launched to load `checkpoints/EXP-MLUA-001_BEST.pth`, that checkpoint was stopped at **Epoch 1 of 200** with a **Validation Dice of 0.000** (66 total optimizer iterations). If executed directly, the Epoch-1 model would predict an entirely blank mask (0 detected pixels).

---

## 9. Important Scientific Notice

> [!WARNING]
> **Scientific & Clinical Status**:  
> The predictions currently rendered in the UI are **client-side procedural simulations** designed for clinical workflow evaluation. They do **NOT** represent the output of a fully trained, converged MLUA deep neural network.  
> 
> The checkpoint on disk (`EXP-MLUA-001_BEST.pth`) represents an **Epoch-1 development snapshot (1/200 epochs)** and must not be presented as a validated clinical caries detection model. Full multi-epoch GPU training and formal validation across the DC1000 cohort remain prerequisites before genuine model inference can be enabled.

---

## 10. Audit Sign-Off Matrix

* **Training Completed**: **NO** (1 / 200 epochs)
* **Checkpoint Path on Disk**: `c:\Users\devin\MLUA\checkpoints\EXP-MLUA-001_BEST.pth`
* **Checkpoint Epoch**: **`1`** (Global Step: `66`)
* **Frontend Uses Real Live Inference**: **NO** (Uses client-side Canvas procedural generator via fallback)
* **Mock / Simulated Data Detected**: **YES** (Procedural client-side overlays + static literature benchmark metrics)
* **Sealed 100-Case Benchmark Set Accessed**: **NO** (Strictly isolated and untouched)
* **Clinical Diagnostic Validity**: **NO** (Development & UI simulation only)
