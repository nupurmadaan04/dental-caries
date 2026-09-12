# Frontend Application & API Architecture

## 1. Architecture Stack

The Dental Caries Clinical AI platform is engineered as a modern, high-performance web application tailored for clinical dental environments:

- **Frontend Framework:** React 18 with TypeScript and Vite.
- **Styling & Design System:** Tailwind CSS with custom medical color tokens, dark/light theme switching, and glassmorphic elevations.
- **Icons & Visuals:** Lucide React icons.
- **Canvas Rendering Engine:** Custom multi-layer HTML5 canvas overlay engine supporting raw radiograph display, AI segmentation masks, ground truth comparison, uncertainty heatmaps, and lesion site bounding boxes.
- **Client-Side PDF Generation:** High-resolution vector-quality 2-page clinical report generator built with `jspdf` and `html2canvas`.

---

## 2. Interactive Analysis Capabilities

1. **Multi-Layer Radiograph Viewer:**
   - Layer 1: Grayscale Original Panoramic Radiograph ($768 \times 1536$).
   - Layer 2: Color-coded AI Segmentation Mask with adjustable opacity.
   - Layer 3: Ground Truth mask overlay for academic auditing and verification.
   - Layer 4: Interactive bounding boxes and numbered lesion site tags ($L_1, L_2, \dots, L_N$).
   - Layer 5: Monte Carlo Epistemic Uncertainty Heatmap.

2. **Real-Time Dynamic Threshold Slider:**
   - Allows clinicians to adjust the detection sensitivity $\tau \in [0.10, 0.90]$ in real-time, instantly recalculating lesion boundaries, pixel counts, and severity stages.

3. **Color-Coded Clinical Assessment Banner:**
   - Dynamically highlights overall case assessment in Green (Stage 0), Yellow (Stage 1), Orange (Stage 2), or Red (Stage 3) with matching glowing indicators and diagnostic summaries.

---

## 3. REST API & Local Fallback Architecture

The frontend interfaces with the MLUA inference service through [`frontend/src/services/api.ts`](file:///c:/Users/devin/MLUA/frontend/src/services/api.ts):

- `POST /api/analyze`: Submits a panoramic radiograph for full 21-patch sliding-window inference and returns segmented coordinates, lesion clusters, uncertainty maps, and stage classifications.
- `GET /api/health`: Polls inference backend status and active model checkpoint (`EXP-MLUA-003_E56_FINAL.pth`).
- **Autonomous Standalone Fallback:** When running without a live GPU backend, the client seamlessly falls back to high-fidelity anatomical mock datasets covering healthy, mild, moderate, and severe clinical cases.

---

## 4. Running the Web Application

```bash
# Navigate to the frontend directory
cd frontend

# Install dependencies
npm install

# Start local development server
npm run dev

# Build optimized production bundle
npm run build
```
