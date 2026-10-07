# MLUA Documentation Index

Welcome to the comprehensive documentation suite for the **MLUA Dental Caries Segmentation and Clinical Intelligence System**.

This directory contains technical specifications, architectural blueprints, clinical evaluation guidelines, benchmark logs, and natural language understanding documentation.

---

## Master Document Directory

| Document | Description | Target Audience |
| :--- | :--- | :--- |
| [**System Architecture**](file:///c:/Users/devin/MLUA/docs/ARCHITECTURE.md) | High-level system design, data flow, MLUA dual-model architecture, backend FastAPI architecture, and frontend React component topology. | Engineers, Architects |
| [**API & Frontend Guide**](file:///c:/Users/devin/MLUA/docs/API_AND_FRONTEND.md) | Full REST and streaming endpoint reference (`/api/chat`, `/api/health`), request/response schemas, frontend state management, and mock mode operation. | Full-Stack Developers |
| [**Clinical Guidelines & Staging**](file:///c:/Users/devin/MLUA/docs/CLINICAL_GUIDELINES.md) | Clinical taxonomy, 3-level application caries staging, lesion classification criteria, diagnostic safety guardrails, and radiologist disclaimers. | Clinicians, Researchers |
| [**Experimental Records & Benchmarks**](file:///c:/Users/devin/MLUA/docs/EXPERIMENTS.md) | Complete experimental progression from supervised baseline (`EXP-MLUA-001`) to canonical production model (`EXP-MLUA-003_E75_BEST.pth`), ablation studies, and Monte Carlo sampling benchmarks. | ML Engineers, Researchers |
| [**NLU Conversational Dataset (500 Rows)**](file:///c:/Users/devin/MLUA/docs/NLU_CONVERSATIONAL_DATASET.md) | Complete documentation of the 500-example conversational QA benchmark dataset, 15 clinical categories, bilingual English/Hinglish distribution, and test suite verification. | NLP Engineers, Researchers |
| [**NLU Dataset & Classifier Pipeline**](file:///c:/Users/devin/MLUA/docs/NLU_DATASET_AND_CLASSIFIER.md) | Technical architecture of the 3-layer NLU router, 31 canonical intent classes, emotion analysis engine, and hybrid fallback inference. | NLP Engineers |
| [**Research Paper Audit**](file:///c:/Users/devin/MLUA/docs/RESEARCH_PAPER_AUDIT.md) | Comparative analysis against the reference MLUA publication, validating methodology alignment, hyperparameter fidelity, and evaluation protocol adherence. | Researchers, Reviewers |
| [**Final Project Capstone Report**](file:///c:/Users/devin/MLUA/docs/FINAL_PROJECT_REPORT.md) | Executive summary and comprehensive capstone report detailing engineering methodologies, quantitative findings, and clinical impact. | Mentors, Stakeholders |
| [**Dataset & Model Preparation**](file:///c:/Users/devin/MLUA/docs/research/FINAL_DATASET_MODEL_MENTOR_PREPARATION.md) | Detailed verification notes on dataset provenance, DC1000 dataset partition schemas, and model checkpoint immutability. | Mentors, Reviewers |

---

## Canonical Model & System Identifiers
- **Active Production Model**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E75_BEST.pth`
  - SHA-256: `cb52ed6ec6911fab7f0081588fb63a6f0bad0e37b61c37f0ba9bd293a6bcedfb`
  - Validation Dice: **71.867%** (operating threshold $\tau = 0.50$)
  - Sealed Test Set Macro Dice: **50.147%** (100 panoramic radiographs)
- **Historical Reference Baseline**: `outputs/experiments/EXP-MLUA-003_FINAL/checkpoints/EXP-MLUA-003_E64_BEST.pth`
  - SHA-256: `167918b8375815668b5e784902e3e82a865b86053a610a312c775d52d473c526`
- **Conversational NLU Dataset**: `backend/nlp/data/conversational_qa_dataset.jsonl` / `.csv` (500 rows, 250 EN / 250 Hinglish)
- **Original Intent Benchmark**: `backend/nlp/data/intent_dataset.jsonl` / `.csv` (250 rows preserved untouched)
