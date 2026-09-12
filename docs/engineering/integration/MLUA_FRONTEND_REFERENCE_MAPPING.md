# MLUA FRONTEND REFERENCE MAPPING & FORENSIC INVENTORY

**Project**: Multi-Level Uncertainty-Aware Semi-Supervised Dental Panoramic Caries Segmentation (MLUA)  
**Document**: `docs/MLUA_FRONTEND_REFERENCE_MAPPING.md`  
**Purpose**: Map visual/UX reference patterns and component architecture to the standalone MLUA frontend implementation.

---

## 1. Reference Architecture & Forensic Inventory

The reference dental panoramic caries AI interface is established from the clinical-research dashboard UX standard, incorporating an **obsidian dark-mode aesthetic**, **multi-level uncertainty heatmaps**, **panoramic image viewer with synchronized overlays**, and **strict clinical decision-support disclaimers**.

### Frontend Technology Stack
- **Framework**: React 18.2+ with TypeScript 5.3+
- **Build Tool**: Vite 5.1+
- **Routing**: React Router DOM v6.22+
- **Styling**: TailwindCSS v3.4+ with custom obsidian & medical brand palette
- **Icons**: Lucide-React
- **State & Client**: React Hooks, Context API (`ThemeContext`), Axios HTTP client with Mock fallback
- **Typography**: Inter font family (Google Fonts)

---

## 2. File-by-File Reference Mapping Table

| Reference Component / Pattern | Reference Purpose | Visual & UX Pattern | MLUA Replacement File | Action |
|---|---|---|---|---|
| `Sidebar.tsx` | Main Navigation | Obsidian dark collapsible sidebar with active route glow, icon indicators, and system badge | `frontend/src/components/Sidebar.tsx` | Rebuild / Adapt |
| `Navbar.tsx` | Top Bar & Utilities | Sticky header with backend health badge, mock mode indicator, quick search, and theme toggle | `frontend/src/components/Navbar.tsx` | Rebuild / Adapt |
| `MetricCard.tsx` | KPI & Parameter Display | Rounded-2xl card with micro-trend badge, glowing icon container, and sub-metric breakdown | `frontend/src/components/MetricCard.tsx` | Rebuild / Adapt |
| `StagingBadge.tsx` | Severity / Caries Staging | Enamel (E1/E2), Dentin (D1/D2/D3), and Deep caries colored pill badges | `frontend/src/components/StagingBadge.tsx` | Rebuild / Adapt |
| `ImageLightboxModal.tsx` | Panoramic Inspection | Full-screen interactive viewer with pan, zoom (1x-5x), split comparison slider, and overlay toggles | `frontend/src/components/ImageLightboxModal.tsx` | Rebuild / Adapt |
| `LesionTable.tsx` | Candidate Region Table | Tabular lesion list with confidence bars, bounding box / polygon stats, and tooth quadrant tags | `frontend/src/components/LesionTable.tsx` | Rebuild / Adapt |
| `ArchitectureDiagram.tsx` | MLUA Network Topology | Interactive visual diagram showing ResNet-34 Encoder $\to$ FPN $\to$ 4 Aux + 1 Fused Heads | `frontend/src/components/ArchitectureDiagram.tsx` | Rebuild / Adapt |
| `UncertaintyHeatmap.tsx` | Entropy & Variance Vis | Dual-view OPG with overlaid Shannon entropy colormap ($H(Y\|X)$) and certainty threshold filter | `frontend/src/components/UncertaintyHeatmap.tsx` | Rebuild / Adapt |
| `StudentTeacherComparison.tsx` | SSL Dual-Stream View | Side-by-side comparison of Student logits vs. EMA Teacher smoothed predictions | `frontend/src/components/StudentTeacherComparison.tsx` | Rebuild / Adapt |
| `MCLoader.tsx` | Monte Carlo Step Animation | Multi-stage radar/progress visualization showing $T=8$ stochastic passes and entropy gating | `frontend/src/components/MCLoader.tsx` | Rebuild / Adapt |
| `ResearchStatusBadge.tsx` | Verification & Gate Status | Distinctive pill badge for Verified Baseline vs. In-Progress vs. Mock Inference | `frontend/src/components/ResearchStatusBadge.tsx` | Rebuild / Adapt |
| `AIAssistantDrawer.tsx` | Research Co-pilot Drawer | Collapsible slide-over drawer explaining MLUA predictions, entropy metrics, and limitations | `frontend/src/components/AIAssistantDrawer.tsx` | Rebuild / Adapt |
| `ThemeContext.tsx` | Theme State Management | Persistent dark/light mode provider with Tailwind `dark` class binding | `frontend/src/context/ThemeContext.tsx` | Rebuild / Adapt |
| `DashboardPage.tsx` | Research Home | System overview, model architecture KPIs, quick analysis launcher, and experiment status | `frontend/src/pages/DashboardPage.tsx` | Rebuild / Adapt |
| `NewAnalysisPage.tsx` | OPG Upload & Inference | Multi-format drag-and-drop uploader with DICOM/PNG support, threshold sliders, and real-time steps | `frontend/src/pages/NewAnalysisPage.tsx` | Rebuild / Adapt |
| `AnalysisResultPage.tsx` | Segmentation & Findings | Complete findings view with tabbed overlays (Clinical, Uncertainty, Candidates, Clinician Notes) | `frontend/src/pages/AnalysisResultPage.tsx` | Rebuild / Adapt |
| `UncertaintyExplorerPage.tsx`| Interactive Entropy Lab | Dynamic slider-based exploration of Monte Carlo passes ($T$), noise $\sigma$, and gating $\gamma$ | `frontend/src/pages/UncertaintyExplorerPage.tsx` | Rebuild / Adapt |
| `HistoryPage.tsx` | Analysis Archive | Searchable, filterable audit log of past OPG analyses with export capabilities | `frontend/src/pages/HistoryPage.tsx` | Rebuild / Adapt |
| `ReportsPage.tsx` | PDF & JSON Reports | Structured clinical research report generator with download & preview actions | `frontend/src/pages/ReportsPage.tsx` | Rebuild / Adapt |
| `MLUAMethodologyPage.tsx` | Mathematical & Algorithmic Guide | Interactive deep-dive into FPN multi-scale supervision, Shannon entropy, and EMA dynamics | `frontend/src/pages/MLUAMethodologyPage.tsx` | Rebuild / Adapt |
| `TechnicalVerificationPage.tsx`| Integrity & Gate Audit | Reproducibility dashboard showing frozen config snapshot, hash audits, and training history | `frontend/src/pages/TechnicalVerificationPage.tsx`| Rebuild / Adapt |
| `LimitationsPage.tsx` | Clinical Safety Disclaimer | Comprehensive medical disclaimer, artifact vulnerability warnings, and OPG distortion notices | `frontend/src/pages/LimitationsPage.tsx` | Rebuild / Adapt |
| `FAQPage.tsx` | Knowledgebase | Searchable FAQ covering SSL theory, thresholding, dental anatomy nuances, and confidence | `frontend/src/pages/FAQPage.tsx` | Rebuild / Adapt |
| `SettingsPage.tsx` | System Configuration | Backend API endpoint manager, mock mode switch, default threshold tuner, and cache purge | `frontend/src/pages/SettingsPage.tsx` | Rebuild / Adapt |

---

## 3. Visual Design Language Extraction

### Color System (Dark Obsidian & Medical Cyan)
```css
/* Background & Surfaces */
--bg-dark: #070a12;
--surface-dark: #0d1322;
--card-dark: #121b2d;
--card-hover: #1a243c;
--border-dark: #1b2742;

/* Brand & Accent */
--brand-500: #0ea5e9;
--brand-400: #38bdf8;
--brand-600: #0284c7;
--accent-purple: #8b5cf6;
--accent-amber: #f59e0b;
--accent-emerald: #10b981;
--accent-rose: #f43f5e;

/* Typography */
--text-primary: #f8fafc;
--text-secondary: #94a3b8;
--text-muted: #64748b;
```

### UI Component Standards
- **Cards**: `bg-[#121b2d]/80 backdrop-blur-md border border-[#1b2742] rounded-2xl shadow-xl`
- **Buttons**: `rounded-xl font-medium transition-all duration-200 shadow-md`
- **Badges**: `rounded-full px-3 py-1 text-xs font-semibold uppercase tracking-wider`
- **Typography**: Inter sans-serif, crisp high-contrast headings with subtle gradient accents
- **Navigation**: Sidebar width `260px` (collapsed: `80px`), Topbar height `64px`
