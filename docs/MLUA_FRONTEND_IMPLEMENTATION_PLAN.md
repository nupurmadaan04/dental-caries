# MLUA FRONTEND IMPLEMENTATION PLAN
## Standalone Panoramic Dental Caries AI Research Frontend

**Project**: Multi-Level Uncertainty-Aware Semi-Supervised Learning for Dental Panoramic Caries Segmentation (MLUA)  
**Status**: Approved for Standalone Implementation  
**Output Path**: `frontend/` (Standalone Vite + React + TypeScript + TailwindCSS)

---

## 1. Architecture Summary
The MLUA Frontend is an interactive, dark-obsidian medical AI research portal engineered specifically for panoramic radiograph (OPG) caries segmentation. It interfaces seamlessly with the MLUA multi-level uncertainty model while maintaining 100% standalone reliability through intelligent Mock/Live dual-mode orchestration.

---

## 2. Directory Structure (`frontend/`)

```
frontend/
├── index.html
├── package.json
├── vite.config.ts
├── tsconfig.json
├── tsconfig.node.json
├── tailwind.config.js
├── postcss.config.js
├── public/
│   ├── favicon.svg
│   └── samples/
│       ├── sample_opg_01.jpg
│       ├── sample_opg_02.jpg
│       └── sample_opg_03.jpg
└── src/
    ├── index.css
    ├── App.tsx
    ├── main.tsx
    │
    ├── components/
    │   ├── Sidebar.tsx
    │   ├── Navbar.tsx
    │   ├── MetricCard.tsx
    │   ├── StagingBadge.tsx
    │   ├── ImageLightboxModal.tsx
    │   ├── LesionTable.tsx
    │   ├── ArchitectureDiagram.tsx
    │   ├── UncertaintyHeatmap.tsx
    │   ├── StudentTeacherComparison.tsx
    │   ├── MCLoader.tsx
    │   ├── ResearchStatusBadge.tsx
    │   └── AIAssistantDrawer.tsx
    │
    ├── context/
    │   └── ThemeContext.tsx
    │
    ├── pages/
    │   ├── DashboardPage.tsx
    │   ├── NewAnalysisPage.tsx
    │   ├── AnalysisResultPage.tsx
    │   ├── UncertaintyExplorerPage.tsx
    │   ├── HistoryPage.tsx
    │   ├── ReportsPage.tsx
    │   ├── MLUAMethodologyPage.tsx
    │   ├── TechnicalVerificationPage.tsx
    │   ├── LimitationsPage.tsx
    │   ├── FAQPage.tsx
    │   └── SettingsPage.tsx
    │
    ├── services/
    │   └── api.ts
    │
    ├── types/
    │   └── api.ts
    │
    ├── constants/
    │   └── mluaMetadata.ts
    │
    └── data/
        └── mockData.ts
```

---

## 3. Route Mapping Table

| Route Path | Component | Primary Objective |
|---|---|---|
| `/` | `DashboardPage` | Executive model summary, verified configuration KPIs, experiment status, quick launch |
| `/analyze` | `NewAnalysisPage` | Multi-format OPG upload (PNG/JPG/DICOM), threshold sliders, 10-step progress simulation |
| `/results/:id` | `AnalysisResultPage` | Panoramic viewer, segmentation masks, uncertainty heatmaps, tooth candidate lesion table |
| `/uncertainty` | `UncertaintyExplorerPage` | Interactive Shannon entropy laboratory, MC variance sliders, certainty gate demonstration |
| `/history` | `HistoryPage` | Audit archive of previous OPG analyses with search, filter, and comparison actions |
| `/reports` | `ReportsPage` | Clinical research report generator with PDF preview and JSON dataset exports |
| `/methodology` | `MLUAMethodologyPage` | Interactive visual guide to ResNet-34 FPN, multi-scale deep supervision, and EMA teacher |
| `/verification` | `TechnicalVerificationPage` | Baseline configuration snapshot, checkpoint verification, and reproducibility audit |
| `/limitations` | `LimitationsPage` | Academic research disclaimer, clinical safety guidelines, and radiographic artifact warnings |
| `/faq` | `FAQPage` | Knowledgebase explaining semi-supervised learning, Monte Carlo sampling, and entropy |
| `/settings` | `SettingsPage` | Backend API URL manager, Live vs Mock toggle, decision threshold presets |

---

## 4. Mock-Data & Dual-Mode API Strategy
1. **Health Check (`GET /api/health`)**:
   - Checks backend connectivity. If unreachable, gracefully sets `isLiveBackend = false` and activates `MOCK_MODE` with clear amber visual indicators.
2. **Analysis (`POST /api/analyze`)**:
   - In Live Mode: Sends multipart form data to MLUA inference server.
   - In Mock Mode: Executes realistic 10-step progress animation (~3.5s) and returns high-fidelity simulated panoramic caries segmentation findings with tooth-by-tooth candidate lesion metadata.
3. **Zero Fabrication Policy**:
   - Unverified benchmark numbers or in-progress training states are explicitly labeled as *"Research Configuration"* or *"In Training / Not Yet Benchmarked"*.

---

## 5. Implementation Sequence
1. **Initialize Standalone Project**: Scaffold `frontend/` with Vite, React 18, TypeScript, TailwindCSS, Lucide-React.
2. **Configure Design System**: Set up obsidian dark palette, custom scrollbars, glassmorphism utilities in `tailwind.config.js` and `src/index.css`.
3. **Core Types & Constants**: Implement `types/api.ts` and `constants/mluaMetadata.ts` with authentic MLUA configurations.
4. **Mock Data & API Layer**: Implement `data/mockData.ts` and `services/api.ts`.
5. **Context & Layout**: Implement `ThemeContext.tsx`, `Sidebar.tsx`, and `Navbar.tsx`.
6. **Reusable Components**: Build `MetricCard`, `StagingBadge`, `ImageLightboxModal`, `LesionTable`, `ArchitectureDiagram`, `UncertaintyHeatmap`, `StudentTeacherComparison`, `MCLoader`, `ResearchStatusBadge`, and `AIAssistantDrawer`.
7. **Implement All 11 Pages**: Build out the complete feature set across all routes.
8. **Build & Validation**: Run `npm run build` to ensure 0 TypeScript / lint errors.
9. **Launch Dev Server**: Start `npm run dev` and verify localhost URL and UI responsiveness.
