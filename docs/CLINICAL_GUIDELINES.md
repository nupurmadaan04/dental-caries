# Clinical Staging & Diagnostic Guidelines

## 1. 4-Tier Color-Coded Severity Staging Matrix

To assist dental clinicians during radiographic review, detected candidate caries regions are categorized into 4 clinical severity stages based on lesion depth, pixel area count, and percentage demineralization:

| Clinical Stage | Color Code | Status Indicator | Pixel Area Threshold | Lesion Extent & Pathological Classification |
| :--- | :--- | :--- | :--- | :--- |
| **Stage 0** | 🟢 **Green** (`emerald`) | **No Caries Detected** | $0\text{ px}$ ($0.00\%$) | Sound, intact enamel and dentin structures without identifiable radiolucency. |
| **Stage 1** | 🟡 **Yellow** (`yellow`) | **Early Demineralization** | $20 - 250\text{ px}$ ($< 0.80\%$) | Enamel-limited micro-lesion ($E_1/E_2$). Non-cavitated incipient demineralization amenable to remineralization therapy. |
| **Stage 2** | 🟠 **Orange** (`orange`) | **Moderate Caries** | $250 - 600\text{ px}$ ($0.80\% - 1.80\%$) | Middle dentin penetration ($D_1/D_2$). Structural breakdown requiring restorative intervention. |
| **Stage 3** | 🔴 **Red** (`red`) | **Extensive Caries** | $> 600\text{ px}$ ($> 1.80\%$) | Deep dentin involvement approaching or penetrating dental pulp ($D_3$). Risk of pulpal necrosis; immediate restorative or endodontic therapy. |

---

## 2. Radiographic Considerations in Panoramic Screening

Panoramic radiography (OPG) compresses complex 3D curved anatomical dental arches into a single 2D plane. Clinicians must apply the following guidelines when evaluating AI overlays:

### A. Interproximal Surfaces
Subtle enamel demineralization on adjacent premolar and molar contact surfaces should always be cross-referenced with intraoral bitewing radiographs for definitive confirmation.

### B. Cervical Burnout Differentiation
The physiological anatomical narrowing of tooth crowns between the enamel cap and alveolar crest naturally creates areas of decreased radiopacity ("cervical burnout"). Clinicians should correlate radiolucencies in cervical zones with tactile probe examination.

### C. Restorative Margins
Composite resin restorations lacking heavy radiopaque fillers may appear radiolucent under automated segmentation. Cross-reference candidate clusters with patient dental treatment histories.

---

## 3. Standardized 2-Page PDF Radiology Report Architecture

The system compiles screening findings into an official 2-page Clinical Radiology Screening & Candidate Localization Report:

- **Page 1 (Screening Overview & Panoramic Visuals):**
  - Clinical header with unique Case ID, modality, operating threshold ($\tau = 0.50$), and date.
  - Panoramic radiograph viewport ($16:9$ natural aspect ratio) with high-contrast lesion overlays and $L_1, L_2, \dots$ badge markers.
  - 4-Card Primary KPI Grid: Overall Case Assessment, Total Detected Sites, Total Demineralized Area, and Mean Detection Confidence.
  - Candidate Region Localization Table detailing each lesion's site ID, coordinates, area in pixels, area percentage, and assigned clinical stage.

- **Page 2 (Lesion Staging, Academic Verification & Clinician Sign-Off):**
  - Visual Caries Staging & Pixel Area Reference Matrix (Stages 0–3).
  - 3-Card Technical Performance KPI Grid: Validation Overlap ($65.62\%$), Jaccard IoU ($49.85\%$), and Specificity ($99.63\%$).
  - Clinical Interpretation Guidelines (Interproximal, Cervical Burnout, Restorative Margins).
  - Expanded Legal Disclaimer & Clinician Review & Sign-Off box for formal medical records archiving.
