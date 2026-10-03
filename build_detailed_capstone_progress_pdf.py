import os
import sys
import pandas as pd
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class DetailedProgressNumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas for dynamic total page numbering,
    running header and running footer on the comprehensive capstone progress report.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        if self._pageNumber == 1:
            # Suppress headers/footers on the cover page
            return

        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4A5568"))

        # Running Header
        header_text = "Capstone Project Progress Report | Dental Caries Segmentation (EXP-MLUA-003)"
        self.drawString(54, A4[1] - 36, header_text)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.6)
        self.line(54, A4[1] - 42, A4[0] - 54, A4[1] - 42)

        # Running Footer
        footer_left = "Student: Nupur Madaan | B.Tech CSE (Academic Year 2026-27)"
        footer_right = f"Page {self._pageNumber} of {page_count}"
        self.drawString(54, 34, footer_left)
        self.drawRightString(A4[0] - 54, 34, footer_right)
        self.line(54, 46, A4[0] - 54, 46)

        self.restoreState()


def create_detailed_progress_pdf(output_filename="Detailed_Capstone_Project_Progress_Report.pdf"):
    print(f"Compiling Detailed Capstone Progress Report: {output_filename}...")
    out_dir = os.path.join("outputs", "progress_figures")
    
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1A365D")   # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0") # Slate Blue
    c_dark = colors.HexColor("#2D3748")      # Charcoal Body Text
    c_light_bg = colors.HexColor("#F7FAFC")  # Light panel bg
    c_border = colors.HexColor("#CBD5E0")    # Border grey
    c_alert_bg = colors.HexColor("#EDF2F7")  # Notice box

    # Typography Styles
    title_style = ParagraphStyle(
        'ProgTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=23,
        textColor=c_primary,
        alignment=1,
        spaceAfter=12
    )

    subtitle_style = ParagraphStyle(
        'ProgSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        leading=15,
        textColor=c_secondary,
        alignment=1,
        spaceAfter=18
    )

    h1_style = ParagraphStyle(
        'ProgH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=15,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'ProgH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=9.8,
        leading=13,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'ProgBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.2,
        leading=11.5,
        textColor=c_dark,
        spaceAfter=4,
        alignment=4 # Justified
    )

    bullet_style = ParagraphStyle(
        'ProgBullet',
        parent=body_style,
        leftIndent=12,
        firstLineIndent=-8,
        spaceAfter=2.5,
        alignment=4
    )

    caption_style = ParagraphStyle(
        'ProgCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.2,
        leading=9.5,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=3,
        spaceAfter=7
    )

    table_header_style = ParagraphStyle(
        'ProgTH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'ProgTD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=6.8,
        leading=8.8,
        textColor=c_dark,
        alignment=0
    )

    table_cell_bold = ParagraphStyle(
        'ProgTDBold',
        parent=table_cell_style,
        fontName='Helvetica-Bold'
    )

    table_cell_center = ParagraphStyle(
        'ProgTDCenter',
        parent=table_cell_style,
        alignment=1
    )

    story = []

    # ==========================================
    # COVER PAGE
    # ==========================================
    story.append(Spacer(1, 15))
    story.append(Paragraph("B.TECH CAPSTONE PROJECT PROGRESS REPORT", ParagraphStyle('TopTag', fontName='Helvetica-Bold', fontSize=10, textColor=c_secondary, alignment=1, spaceAfter=8)))
    story.append(HRFlowable(width="100%", thickness=2, color=c_primary, spaceAfter=15))
    
    story.append(Paragraph("Progress Report on Dental Caries Segmentation from Panoramic Dental X-Ray Images Using Multi-Level Uncertainty-Aware Learning", title_style))
    story.append(Paragraph("Comprehensive Experimental Journey: Supervised Baselines (EXP001-EXP006), MLUA Exploration (EXP-MLUA-001/002), Forensic Buffer Remediation, and 60-Epoch Benchmark Validation (EXP-MLUA-003)", subtitle_style))
    
    story.append(Spacer(1, 20))

    meta_table = [
        [Paragraph("<b>Candidate Name:</b>", table_cell_bold), Paragraph("Nupur Madaan", table_cell_style)],
        [Paragraph("<b>Degree Program:</b>", table_cell_bold), Paragraph("Bachelor of Technology (B.Tech) in Computer Science & Engineering", table_cell_style)],
        [Paragraph("<b>Enrollment / Roll No.:</b>", table_cell_bold), Paragraph("[Placeholder: Student Enrollment Number]", table_cell_style)],
        [Paragraph("<b>Department:</b>", table_cell_bold), Paragraph("[Placeholder: Department of Computer Science & Engineering]", table_cell_style)],
        [Paragraph("<b>Institution / University:</b>", table_cell_bold), Paragraph("[Placeholder: University / Institute Name]", table_cell_style)],
        [Paragraph("<b>Faculty Guide / Supervisor:</b>", table_cell_bold), Paragraph("[Placeholder: Faculty Supervisor / Guide Name]", table_cell_style)],
        [Paragraph("<b>Project Domain:</b>", table_cell_bold), Paragraph("Medical Image Segmentation & Semi-Supervised Deep Learning", table_cell_style)],
        [Paragraph("<b>Current Experimental Milestone:</b>", table_cell_bold), Paragraph("EXP-MLUA-003 (Selected Checkpoint: EXP-MLUA-003_E56_FINAL.pth)", table_cell_style)],
        [Paragraph("<b>Evaluated Benchmark:</b>", table_cell_bold), Paragraph("DC1000 Panoramic Radiograph Dataset (1,000 Cases)", table_cell_style)],
        [Paragraph("<b>Academic Year:</b>", table_cell_bold), Paragraph("2026-27", table_cell_style)],
    ]
    t_meta = Table(meta_table, colWidths=[150, 310])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light_bg),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3.5),
        ('TOPPADDING', (0,0), (-1,-1), 3.5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('LINEBELOW', (0,0), (-1,-2), 0.5, c_border),
    ]))
    story.append(t_meta)
    
    story.append(Spacer(1, 20))

    notice_box = [
        [Paragraph("<b>Progress Narrative Notice:</b> This capstone progress report presents the complete research and engineering trajectory of our dental caries segmentation project. It explicitly documents our early supervised experiments (EXP001-EXP006), the initial MLUA semi-supervised trials, the numerical instability and NaN collapse in EXP-MLUA-002, the deep forensic investigation identifying missing Teacher BatchNorm buffer synchronization, the implementation remediation, the completed 60-epoch EXP-MLUA-003 benchmark (Val Dice = 65.62%, Sealed Test Macro Dice = 43.04%), and the remaining engineering deliverables.", ParagraphStyle('NoticeP', parent=body_style, fontSize=7.5, leading=10.5))]
    ]
    t_notice = Table(notice_box, colWidths=[480])
    t_notice.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_alert_bg),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
    ]))
    story.append(t_notice)
    story.append(PageBreak())

    # ==========================================
    # TABLE OF CONTENTS
    # ==========================================
    story.append(Paragraph("TABLE OF CONTENTS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    toc_rows = [
        ("1. ABSTRACT", "3"),
        ("2. INTRODUCTION", "3"),
        ("3. OBJECTIVES", "4"),
        ("4. PROBLEM STATEMENT", "4"),
        ("5. PROJECT BACKGROUND AND PAPER REFERENCE", "5"),
        ("6. DATASET DESCRIPTION (DC1000 BENCHMARK)", "5"),
        ("7. PROJECT PROGRESS & MILESTONE JOURNEY", "6"),
        ("8. OVERALL SYSTEM FLOWCHART", "7"),
        ("9. DETAILED METHODOLOGY", "8"),
        ("    9.1 Input Collection & Modality", "8"),
        ("    9.2 Dataset Partitioning (20% Labeled / 80% Unlabeled)", "8"),
        ("    9.3 Annotation & Ground Truth Scope", "8"),
        ("    9.4 Data Cleaning & Normalization", "8"),
        ("    9.5 384x384 Patch Extraction (Stride S=192)", "9"),
        ("    9.6 ResNet-34 Feature Extraction Backbone", "9"),
        ("    9.7 Feature Pyramid Network (FPN) Decoder", "9"),
        ("    9.8 Multi-Scale Auxiliary & Fused Segmentation Heads", "9"),
        ("    9.9 Teacher-Student Consistency Architecture", "10"),
        ("    9.10 Loss Formulations (BCE + Soft Dice + Consistency MSE)", "10"),
        ("    9.11 Teacher Exponential Moving Average (EMA) Update", "10"),
        ("    9.12 Monte Carlo Sampling (T=8) & Uncertainty Estimation", "10"),
        ("    9.13 Dynamic Confidence Masking", "11"),
        ("    9.14 Sliding-Window Full OPG Reconstruction", "11"),
        ("    9.15 Operating Threshold Selection", "11"),
        ("    9.16 Final Output Representation", "11"),
        ("10. MODULAR SYSTEM IMPLEMENTATION", "12"),
        ("11. EXPERIMENTAL PROGRESS AND COMPLETE TRIAL HISTORY", "13"),
        ("    11.1 Supervised Baseline Trials (EXP001 to EXP006)", "13"),
        ("    11.2 EXP-MLUA-001 (10% SSL Baseline & Foreground Suppression)", "14"),
        ("    11.3 EXP-MLUA-002 (Consistency Scaling & Numerical Collapse)", "15"),
        ("    11.4 Forensic Root Cause Analysis of EXP-MLUA-002", "16"),
        ("    11.5 Implementation Remediation: Dual Parameter & Buffer EMA Sync", "17"),
        ("    11.6 EXP-MLUA-003 (Corrected 60-Epoch Benchmark Run)", "18"),
        ("    11.7 EXP-MLUA-003 Complete Epoch-by-Epoch History (E1 to E60)", "18"),
        ("    11.8 Milestone Progression & Fluctuation Analysis", "20"),
        ("    11.9 Validation Checkpoint Selection (Epoch 56)", "21"),
        ("    11.10 Validation Threshold Sensitivity Analysis (tau in [0.05, 0.95])", "21"),
        ("    11.11 Independent Sealed Test Evaluation (100 OPG Cases)", "22"),
        ("    11.12 Per-Case Sealed Test Distribution & Outlier Analysis", "23"),
        ("12. COMPREHENSIVE EXPERIMENTAL COMPARISON", "24"),
        ("13. VALIDATION VS. SEALED TEST GENERALIZATION GAP", "25"),
        ("14. CURRENT STATUS SUMMARY", "26"),
        ("15. WORK REMAINING & CAPSTONE DELIVERABLES", "26"),
        ("16. CONCLUSION", "27"),
        ("17. FUTURE RESEARCH SCOPE", "27"),
        ("18. REFERENCES", "28")
    ]

    toc_table_data = []
    for item, pg in toc_rows:
        is_h = not item.startswith("    ")
        st = table_cell_bold if is_h else table_cell_style
        toc_table_data.append([Paragraph(item, st), Paragraph(f"Page {pg}", table_cell_center)])

    t_toc = Table(toc_table_data, colWidths=[400, 80])
    t_toc.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.2),
        ('TOPPADDING', (0,0), (-1,-1), 1.2),
        ('LINEBELOW', (0,0), (-1,-1), 0.3, colors.HexColor("#EDF2F7")),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # ==========================================
    # SECTION 1: ABSTRACT
    # ==========================================
    story.append(Paragraph("1. ABSTRACT", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    
    story.append(Paragraph("Automated semantic segmentation of dental caries on orthopantomograms (panoramic dental X-rays) offers significant potential for computer-aided diagnosis in oral radiology. However, panoramic caries segmentation is fundamentally challenged by severe class imbalance (caries occupies <0.5% of total image canvas), subtle radiolucency with diffuse demineralization margins, anatomical superimpositions (cervical burnout, restoration scatter), and clinical annotation scarcity. To address annotation constraints, this capstone project investigates semi-supervised learning based on the Multi-Level Uncertainty Aware (MLUA) Teacher-Student framework using the benchmark DC1000 dataset (1,000 panoramic radiographs).", body_style))

    story.append(Paragraph("This progress report details our full experimental journey across multiple model iterations. We began by evaluating supervised baselines (EXP001-EXP006), where DeepLabV3+ achieved a peak validation Dice of 48.39% at 512&times;512 resolution. Transitioning to semi-supervised learning, our initial MLUA implementation (EXP-MLUA-001) suffered from foreground suppression and severe metric divergence (Dice dropping from 14.71% at E19 to 0.24% at E22). Scaling consistency loss in EXP-MLUA-002 resulted in catastrophic numerical collapse (NaN loss at Epoch 10, Step 1257). Deep forensic auditing isolated the root cause to unsynchronized Teacher BatchNorm running buffers during EMA updates, causing internal GroupNorm activations to explode (~1.32&times;10<sup>18</sup>). By implementing dual parameter and buffer EMA synchronization, <b>EXP-MLUA-003</b> achieved robust convergence across 60 complete epochs (7,920 steps).", body_style))

    story.append(Paragraph("The optimal checkpoint, selected at <b>Epoch 56</b> (&tau; = 0.50), achieved peak validation metrics: <b>Dice = 65.623%</b>, <b>IoU = 49.854%</b>, <b>Precision = 69.009%</b>, and <b>Recall = 63.649%</b>. Independent evaluation on a sealed test set of 100 full panoramic images achieved <b>Macro Dice = 43.041%</b>, <b>Macro Recall = 52.896%</b>, and <b>Specificity = 99.630%</b> (Micro Dice = 43.391%). The documented generalization gap provides clear direction for our remaining deliverables: morphological false-positive reduction on cervical burnout, low-performing case error taxonomy, and final application interface integration.", body_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 2: INTRODUCTION
    # ==========================================
    story.append(Paragraph("2. INTRODUCTION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))

    story.append(Paragraph("Dental caries is a multifactorial bacterial disease resulting in localized demineralization of hard tooth structures. Panoramic dental radiography (orthopantomography) is the primary diagnostic imaging modality used in dental clinics because a single low-dose scan captures the entire dentomaxillofacial anatomy. However, automated caries segmentation on panoramic X-rays faces severe challenges:", body_style))
    story.append(Paragraph("&bull; <b>Extreme Class Imbalance:</b> Carious lesions constitute a minute fraction (<0.5%) of the full panoramic pixel area.", bullet_style))
    story.append(Paragraph("&bull; <b>Diffuse Demineralization Boundaries:</b> Early lesions present as faint radiolucencies without discrete structural edges.", bullet_style))
    story.append(Paragraph("&bull; <b>Confounding Anatomical Artifacts:</b> Cervical burnout, overlapping proximal surfaces, and ghost images mimic carious radiolucency.", bullet_style))
    story.append(Paragraph("&bull; <b>High Cost of Pixel Annotations:</b> Manual pixel contouring by dental specialists is labor-intensive and costly.", bullet_style))

    story.append(Paragraph("Semi-supervised learning (SSL) provides an effective paradigm by utilizing a small fraction of labeled images (20%) alongside a large volume of unlabeled images (80%). The Multi-Level Uncertainty Aware (MLUA) framework uses a Teacher-Student architecture with Monte Carlo uncertainty estimation to generate reliable pseudo-supervision while filtering out ambiguous boundary predictions. This report chronicles our progressive implementation, troubleshooting, and empirical validation on the DC1000 benchmark.", body_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 3 & 4: OBJECTIVES & PROBLEM STATEMENT
    # ==========================================
    story.append(Paragraph("3. OBJECTIVES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    
    objs = [
        "1. Implement and evaluate baseline supervised segmentation models (DeepLabV3+, FPN, DoubleU-Net) on panoramic radiographs.",
        "2. Implement the semi-supervised Teacher-Student MLUA architecture using ResNet-34 encoder and FPN decoder on the DC1000 dataset.",
        "3. Investigate the causes of foreground degradation in EXP-MLUA-001 and numerical collapse in EXP-MLUA-002.",
        "4. Formulate and implement a dual parameter-and-buffer EMA synchronization mechanism to guarantee Teacher numerical stability.",
        "5. Execute a full 60-epoch training benchmark (EXP-MLUA-003) and log complete epoch-by-epoch validation dynamics.",
        "6. Conduct an internal validation threshold sweep (&tau; &isin; [0.05, 0.95]) to establish the optimal operating cutoff (&tau; = 0.50).",
        "7. Perform unbiased evaluation on an independent sealed test set of 100 full panoramic images using sliding-window reconstruction.",
        "8. Quantify the validation-to-test generalization gap and define concrete technical tasks for capstone completion."
    ]
    for o in objs:
        story.append(Paragraph(o, bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("4. PROBLEM STATEMENT", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    
    story.append(Paragraph("<b>Mathematical Problem Formulation:</b> Given a panoramic dental radiograph <i>X</i> &isin; &real;<sup>H &times; W</sup>, learn a parameterized mapping <i>f<sub>&theta;</sub>(X)</i> producing a continuous pixel-wise probability map <i>P(Y = 1 | X)</i> for dental caries demineralization, and convert it into a discrete binary mask <i>&Ycirc;<sub>&tau;</sub></i> &isin; {0, 1}<sup>H &times; W</sup> via decision threshold &tau; = 0.50. The model must maximize spatial overlap with expert annotations under semi-supervised data constraints (20% labeled, 80% unlabeled) while maintaining numerical stability.", body_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 5 & 6: BACKGROUND & DATASET
    # ==========================================
    story.append(Paragraph("5. PROJECT BACKGROUND AND PAPER REFERENCE", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    story.append(Paragraph("Our project builds upon the foundational research by Wang et al. (<i>Neurocomputing</i>, 2023), entitled <i>\"Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation\"</i>. The published work demonstrated that combining multi-scale feature pyramids with Monte Carlo dropout uncertainty estimation enables effective semi-supervised learning on dental radiographs. We adapted this methodology to modern PyTorch pipelines and evaluated it on the standardized DC1000 dataset.", body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("6. DATASET DESCRIPTION (DC1000 BENCHMARK)", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    story.append(Paragraph("The experiment utilizes the public <b>DC1000</b> dental caries dataset comprising 1,000 digitized panoramic radiographs with over 7,500 expert-annotated carious lesions (593 detailed and 407 rough annotation subsets). In our system, images are standardized to <b>768 &times; 1536 pixels</b> and partitioned into overlapping <b>384 &times; 384 patches</b> (stride S = 192). The dataset is divided into 20% labeled training (530 patches), 80% unlabeled training (1,859 patches), 598 validation patches, and an independent sealed test set of 100 full panoramic images.", body_style))
    story.append(Paragraph("<b>Binary Task Definition:</b> While dataset literature describes clinical severity grades (shallow, middle, deep), our model is strictly formulated for <b>pixel-level binary segmentation</b> (0 = background/healthy, 1 = suspected caries).", body_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 7 & 8: PROGRESS JOURNEY & FLOWCHARTS
    # ==========================================
    story.append(Paragraph("7. PROJECT PROGRESS & MILESTONE JOURNEY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    story.append(Paragraph("The capstone project represents a structured engineering progression across supervised exploration, semi-supervised failure analysis, and benchmark validation.", body_style))

    story.append(Paragraph("8. OVERALL SYSTEM FLOWCHART", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))

    fig1_p = os.path.join(out_dir, "figure1_overall_flowchart.png")
    if os.path.exists(fig1_p):
        story.append(Image(fig1_p, width=480, height=240))
        story.append(Paragraph("<b>Figure 1:</b> Overall system architecture and semi-supervised data flow for EXP-MLUA-003, showing patch extraction, dual network consistency, confidence masking, and sliding-window reconstruction.", caption_style))

    fig2_p = os.path.join(out_dir, "figure2_progress_flowchart.png")
    if os.path.exists(fig2_p):
        story.append(Image(fig2_p, width=480, height=220))
        story.append(Paragraph("<b>Figure 2:</b> Chronological research and experimental progress flowchart from supervised baselines (EXP001-006) through MLUA forensic remediation to final sealed test evaluation.", caption_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 9: METHODOLOGY
    # ==========================================
    story.append(Paragraph("9. DETAILED METHODOLOGY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))

    m_subs = [
        ("9.1 Input Collection & Modality", "Input comprises digitized panoramic dental X-rays (OPGs) capturing bilateral dentomaxillofacial structures in a single exposure."),
        ("9.2 Dataset Partitioning", "Strict semi-supervised split: 20% labeled (530 patches) and 80% unlabeled (1,859 patches), totaling 2,389 training patches. 598 validation patches and 100 sealed test OPGs."),
        ("9.3 Annotation Scope", "Binary ground truth masks (0 = Background/Healthy, 1 = Carious Demineralization) derived from expert clinical consensus."),
        ("9.4 Preprocessing & Normalization", "Images standardized to 768x1536 resolution; min-max scaled and normalized using ImageNet constants (mu = [0.485, 0.456, 0.406], sigma = [0.229, 0.224, 0.225])."),
        ("9.5 Patch Extraction", "Sliding window extraction of 384x384 patches with vertical/horizontal stride S=192 (50% overlap), yielding 21 patches per panoramic image."),
        ("9.6 ResNet-34 Feature Backbone", "Pre-trained ResNet-34 encoder extracts hierarchical multi-scale feature representations across stages C2 (64ch), C3 (128ch), C4 (256ch), and C5 (512ch)."),
        ("9.7 FPN Decoder", "Feature Pyramid Network merges deep semantics with spatial details via lateral 1x1 convs and top-down bilinear upsampling, producing pyramid levels P2-P5 (256 ch)."),
        ("9.8 Auxiliary & Fused Heads", "Four auxiliary 1-channel convolution heads provide deep multi-scale supervision. A primary fused head concatenates all levels for unified 384x384 output."),
        ("9.9 Teacher-Student Framework", "Dual network replicas: Student updated via gradient backpropagation; Teacher updated via Exponential Moving Average (EMA) to provide stable pseudo-supervision."),
        ("9.10 Loss Formulations", "Supervised loss on labeled data: L_sup = L_BCE + L_Dice. Unsupervised consistency loss on unlabeled data: masked Mean Squared Error (MSE) between Student and Teacher."),
        ("9.11 Teacher EMA Update", "Teacher weights updated with momentum alpha = 0.99. Teacher BatchNorm running mean and variance buffers are explicitly synchronized with Student buffers."),
        ("9.12 Monte Carlo Sampling (T=8)", "Teacher performs T=8 stochastic forward passes under active dropout/perturbation per unlabeled patch to compute pixel-wise variance (predictive uncertainty)."),
        ("9.13 Dynamic Confidence Masking", "Binary confidence mask M(i, j) = I(Uncertainty < beta) filters out high-variance boundary pixels from consistency loss calculation."),
        ("9.14 Sliding-Window Reconstruction", "Inference extracts 21 overlapping 384x384 patches, computes continuous probabilities, and reconstructs full 768x1536 probability maps via linear overlap blending."),
        ("9.15 Threshold Selection", "Operating threshold tau = 0.50 selected via validation-only sensitivity sweep across tau in [0.05, 0.95] for balanced precision and recall."),
        ("9.16 Final Output Representation", "Outputs continuous caries probability map, thresholded binary segmentation mask (tau = 0.50), and bounding coordinates highlighting suspected caries candidates.")
    ]
    for title, desc in m_subs:
        story.append(Paragraph(f"<b>{title}:</b> {desc}", body_style))

    fig3_p = os.path.join(out_dir, "figure3_teacher_student.png")
    if os.path.exists(fig3_p):
        story.append(Image(fig3_p, width=440, height=200))
        story.append(Paragraph("<b>Figure 3:</b> Teacher-Student semi-supervised consistency framework with dual parameter & buffer EMA synchronization and Monte Carlo uncertainty estimation.", caption_style))

    fig4_p = os.path.join(out_dir, "figure4_resnet34_fpn.png")
    if os.path.exists(fig4_p):
        story.append(Image(fig4_p, width=440, height=200))
        story.append(Paragraph("<b>Figure 4:</b> ResNet-34 encoder and Feature Pyramid Network (FPN) decoder architecture with lateral connections and multi-scale auxiliary heads.", caption_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 10: IMPLEMENTATION MODULES
    # ==========================================
    story.append(Paragraph("10. MODULAR SYSTEM IMPLEMENTATION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))

    impl_modules = [
        ("10.1 Dataset & Loader Module", "src/mlua/data/dataset.py", "Manages DC1000 image/mask loading, normalization, and simultaneous collation of 4 labeled + 4 unlabeled patches per step."),
        ("10.2 Preprocessing & Patch Module", "src/mlua/data/patch_extractor.py", "Implements sliding-window cropping (384x384, stride 192) and stochastic spatial/intensity augmentations."),
        ("10.3 Model Architecture Module", "src/mlua/models/fpn.py", "Defines ResNet-34 backbone, FPN lateral/top-down pathways, 4 auxiliary heads, and fused segmentation output."),
        ("10.4 Teacher-Student Engine", "src/mlua/models/teacher_student.py", "Orchestrates dual model state dictionaries, EMA parameter updates, and BatchNorm buffer synchronization."),
        ("10.5 Loss & Uncertainty Module", "src/mlua/losses/combined_loss.py", "Computes BCE, Soft Dice, Monte Carlo uncertainty variance (T=8), confidence masks, and consistency MSE."),
        ("10.6 Training & Validation Loop", "src/mlua/trainers/trainer.py", "Executes 60-epoch optimization using AdamW (lr=0.001, wd=0.01) with Cosine Annealing learning rate scheduling."),
        ("10.7 Inference & Visualization", "src/mlua/inference/reconstruction.py", "Stitches patch predictions into full 768x1536 masks, applies threshold tau=0.50, and generates candidate overlays.")
    ]

    t_impl_data = [[Paragraph("<b>Module Name</b>", table_header_style), Paragraph("<b>Implementation File</b>", table_header_style), Paragraph("<b>Core Functional Scope</b>", table_header_style)]]
    for m_name, m_file, m_scope in impl_modules:
        t_impl_data.append([Paragraph(m_name, table_cell_bold), Paragraph(m_file, table_cell_style), Paragraph(m_scope, table_cell_style)])

    t_impl = Table(t_impl_data, colWidths=[120, 140, 220])
    t_impl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t_impl)
    story.append(Paragraph("<b>Table 1:</b> Modular PyTorch implementation architecture across the codebase.", caption_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 11: EXPERIMENTAL PROGRESS & FULL TRIAL HISTORY
    # ==========================================
    story.append(Paragraph("11. EXPERIMENTAL PROGRESS AND COMPLETE TRIAL HISTORY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))

    story.append(Paragraph("11.1 Supervised Baseline Trials (EXP001 to EXP006)", h2_style))
    story.append(Paragraph("Prior to implementing semi-supervised learning, six fully supervised segmentation experiments were conducted to evaluate different architectures, loss functions, and input resolutions. Table 2 summarizes these historical trials.", body_style))

    t_sup_data = [
        [Paragraph("<b>Exp ID</b>", table_header_style), Paragraph("<b>Architecture</b>", table_header_style), Paragraph("<b>Input / Res.</b>", table_header_style), Paragraph("<b>Loss / Config</b>", table_header_style), Paragraph("<b>Validation Dice</b>", table_header_style), Paragraph("<b>Outcome / Decision</b>", table_header_style)],
        [Paragraph("EXP001", table_cell_bold), Paragraph("DeepLabV3+ (ResNet-18)", table_cell_style), Paragraph("Patches (384x384)", table_cell_style), Paragraph("CE + Dice (30 Ep)", table_cell_style), Paragraph("36.12%", table_cell_style), Paragraph("Baseline patch setup established", table_cell_style)],
        [Paragraph("EXP001_EXT", table_cell_bold), Paragraph("DeepLabV3+ (ResNet-18)", table_cell_style), Paragraph("Patches (384x384)", table_cell_style), Paragraph("Continuation (50 Ep)", table_cell_style), Paragraph("39.45%", table_cell_style), Paragraph("Marginal gain; severe boundary noise", table_cell_style)],
        [Paragraph("EXP002", table_cell_bold), Paragraph("DeepLabV3+ (ResNet-18)", table_cell_style), Paragraph("Patches (384x384)", table_cell_style), Paragraph("Focal Loss + Dice", table_cell_style), Paragraph("41.20%", table_cell_style), Paragraph("Focal loss alleviated class imbalance", table_cell_style)],
        [Paragraph("EXP003", table_cell_bold), Paragraph("FPN (ResNet-18)", table_cell_style), Paragraph("Patches (384x384)", table_cell_style), Paragraph("BCE + Dice", table_cell_style), Paragraph("43.85%", table_cell_style), Paragraph("FPN multi-scale fusion improved detail", table_cell_style)],
        [Paragraph("EXP004", table_cell_bold), Paragraph("DeepLabV3+ (ResNet-18)", table_cell_style), Paragraph("Full OPG (384x384)", table_cell_style), Paragraph("BCE + Dice (50 Ep)", table_cell_style), Paragraph("32.40%", table_cell_style), Paragraph("Full downsampling lost small lesions", table_cell_style)],
        [Paragraph("EXP005", table_cell_bold), Paragraph("DoubleU-Net (VGG-19)", table_cell_style), Paragraph("Full OPG (512x512)", table_cell_style), Paragraph("BCE + Dice (50 Ep)", table_cell_style), Paragraph("44.15%", table_cell_style), Paragraph("High parameter count; heavy compute", table_cell_style)],
        [Paragraph("EXP006", table_cell_bold), Paragraph("DeepLabV3+ (ResNet-18)", table_cell_style), Paragraph("Full OPG (512x512)", table_cell_style), Paragraph("BCE+Dice (70 Ep, &tau;=0.10)", table_cell_style), Paragraph("48.39%", table_cell_style), Paragraph("Peak supervised baseline (16.6M params)", table_cell_style)],
    ]
    t_sup = Table(t_sup_data, colWidths=[65, 105, 95, 85, 65, 65])
    t_sup.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t_sup)
    story.append(Paragraph("<b>Table 2:</b> Historical supervised segmentation experiments (EXP001 to EXP006) on the DC1000 dataset.", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("11.2 EXP-MLUA-001 (10% SSL Baseline & Foreground Suppression)", h2_style))
    story.append(Paragraph("<b>EXP-MLUA-001</b> was our initial 10% labeled SSL trial (265 labeled, 2,124 unlabeled patches, ResNet-34 + FPN, seed 42). While early training showed learning, the model suffered from severe foreground suppression beyond Epoch 18: Validation Dice collapsed from 14.71% at E19 to 7.80% at E20, 2.56% at E21, and 0.24% at E22. Maximum foreground probability dropped below 0.75, and predictions at &tau; = 0.50 became essentially zero (Dice = 0.089%), indicating severe over-regularization by unconstrained consistency loss. Table 3 and Figure 5 document this degradation.", body_style))

    fig5_p = os.path.join(out_dir, "figure5_exp001_progression.png")
    if os.path.exists(fig5_p):
        story.append(Image(fig5_p, width=440, height=190))
        story.append(Paragraph("<b>Figure 5:</b> EXP-MLUA-001 training history showing rapid foreground suppression and metric collapse between Epochs 19 and 22.", caption_style))

    t_exp001_data = [
        [Paragraph("<b>Epoch</b>", table_header_style), Paragraph("<b>Train Loss</b>", table_header_style), Paragraph("<b>Val Loss</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>Precision (%)</b>", table_header_style), Paragraph("<b>Recall (%)</b>", table_header_style), Paragraph("<b>Max FG Prob</b>", table_header_style)],
        [Paragraph("Epoch 01", table_cell_bold), Paragraph("0.6652", table_cell_style), Paragraph("1.0521", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.0412", table_cell_style)],
        [Paragraph("Epoch 10", table_cell_bold), Paragraph("0.6511", table_cell_style), Paragraph("1.0345", table_cell_style), Paragraph("1.250%", table_cell_style), Paragraph("12.45%", table_cell_style), Paragraph("0.850%", table_cell_style), Paragraph("0.6840", table_cell_style)],
        [Paragraph("Epoch 15", table_cell_bold), Paragraph("0.6480", table_cell_style), Paragraph("1.0210", table_cell_style), Paragraph("6.840%", table_cell_style), Paragraph("18.20%", table_cell_style), Paragraph("4.350%", table_cell_style), Paragraph("0.7120", table_cell_style)],
        [Paragraph("Epoch 19", table_cell_bold), Paragraph("0.6443", table_cell_style), Paragraph("1.0189", table_cell_style), Paragraph("14.705%", table_cell_style), Paragraph("15.399%", table_cell_style), Paragraph("16.582%", table_cell_style), Paragraph("0.8978", table_cell_style)],
        [Paragraph("Epoch 20", table_cell_bold), Paragraph("0.6429", table_cell_style), Paragraph("1.0127", table_cell_style), Paragraph("7.800%", table_cell_style), Paragraph("36.258%", table_cell_style), Paragraph("5.125%", table_cell_style), Paragraph("0.8008", table_cell_style)],
        [Paragraph("Epoch 21", table_cell_bold), Paragraph("0.6431", table_cell_style), Paragraph("1.0293", table_cell_style), Paragraph("2.555%", table_cell_style), Paragraph("17.153%", table_cell_style), Paragraph("1.390%", table_cell_style), Paragraph("0.8205", table_cell_style)],
        [Paragraph("Epoch 22", table_cell_bold), Paragraph("0.6431", table_cell_style), Paragraph("1.0338", table_cell_style), Paragraph("0.235%", table_cell_style), Paragraph("5.064%", table_cell_style), Paragraph("0.123%", table_cell_style), Paragraph("0.7438", table_cell_style)],
    ]
    t_exp001 = Table(t_exp001_data, colWidths=[65, 70, 70, 70, 70, 65, 70])
    t_exp001.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t_exp001)
    story.append(Paragraph("<b>Table 3:</b> EXP-MLUA-001 validation dynamics showing severe foreground suppression past Epoch 19.", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("11.3 EXP-MLUA-002 (Consistency Scaling & Numerical Collapse)", h2_style))
    story.append(Paragraph("In <b>EXP-MLUA-002</b>, we adjusted loss weights and augmentations. However, the model remained at 0% Dice for Epochs 1-9 due to zero foreground activation. At <b>Epoch 10, Global Step 1257 (Batch 68/132)</b>, training crashed with a non-finite (NaN) loss value during the Teacher forward pass.", body_style))

    fig6_p = os.path.join(out_dir, "figure6_exp002_timeline.png")
    if os.path.exists(fig6_p):
        story.append(Image(fig6_p, width=440, height=190))
        story.append(Paragraph("<b>Figure 6:</b> EXP-MLUA-002 training timeline showing persistent zero-foreground predictions and subsequent NaN collapse at Epoch 10, Step 1257.", caption_style))

    t_exp002_data = [
        [Paragraph("<b>Epoch</b>", table_header_style), Paragraph("<b>Train Loss</b>", table_header_style), Paragraph("<b>Val Loss</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>Precision (%)</b>", table_header_style), Paragraph("<b>Recall (%)</b>", table_header_style), Paragraph("<b>Max FG Prob</b>", table_header_style), Paragraph("<b>Zero-Pred</b>", table_header_style)],
        [Paragraph("Epoch 01", table_cell_bold), Paragraph("0.6634", table_cell_style), Paragraph("1.0493", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.0369", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("Epoch 03", table_cell_bold), Paragraph("0.6549", table_cell_style), Paragraph("1.0436", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.0454", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("Epoch 06", table_cell_bold), Paragraph("0.6531", table_cell_style), Paragraph("1.0418", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.1850", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("Epoch 09", table_cell_bold), Paragraph("0.6517", table_cell_style), Paragraph("1.0384", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.000%", table_cell_style), Paragraph("0.2709", table_cell_style), Paragraph("100.0%", table_cell_style)],
        [Paragraph("Epoch 10*", table_cell_bold), Paragraph("NaN", table_cell_bold), Paragraph("NaN", table_cell_bold), Paragraph("NaN", table_cell_bold), Paragraph("NaN", table_cell_bold), Paragraph("NaN", table_cell_bold), Paragraph("NaN", table_cell_bold), Paragraph("CRASHED", table_cell_bold)],
    ]
    t_exp002 = Table(t_exp002_data, colWidths=[55, 60, 60, 65, 60, 60, 60, 60])
    t_exp002.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t_exp002)
    story.append(Paragraph("<b>Table 4:</b> EXP-MLUA-002 training history (*Training collapsed at Epoch 10, Global Step 1257).", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("11.4 Forensic Root Cause Analysis of EXP-MLUA-002", h2_style))
    story.append(Paragraph("A deep diagnostic audit of checkpoint `EXP-MLUA-002_E10_CRASH` revealed:", body_style))
    story.append(Paragraph("&bull; <b>First Failing Stage:</b> Teacher network forward pass during GroupNorm computation.", bullet_style))
    story.append(Paragraph("&bull; <b>Root Cause:</b> 36/36 Teacher BatchNorm `running_mean` and `running_var` buffers remained at uninitialized default values (`running_var = 1.0`) because Teacher operated in `eval()` mode. Meanwhile, Student `running_var` reached 247.90.", bullet_style))
    story.append(Paragraph("&bull; <b>Activation Explosion:</b> The disparity between EMA-updated weights and uninitialized buffer statistics caused intermediate stage C5 activations to explode to approximately <b>1.32 &times; 10<sup>18</sup></b>, triggering GroupNorm NaN overflow.", bullet_style))
    story.append(Paragraph("&bull; <b>Counterfactual Proof:</b> When Teacher buffers were manually synchronized with Student statistics, peak C5 activations dropped from 1.32&times;10<sup>18</sup> to <b>11.09</b>, producing finite, bounded outputs. Figure 7 highlights these forensic measurements.", bullet_style))

    fig7_p = os.path.join(out_dir, "figure7_bn_statistics.png")
    if os.path.exists(fig7_p):
        story.append(Image(fig7_p, width=440, height=190))
        story.append(Paragraph("<b>Figure 7:</b> Forensic comparison: Teacher vs. Student BatchNorm running statistics and resulting GroupNorm input activation magnitudes.", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("11.5 Implementation Remediation: Dual Parameter & Buffer EMA Synchronization", h2_style))
    story.append(Paragraph("To remediate the root cause, we updated the Teacher EMA routine in `src/mlua/models/teacher_student.py` to synchronize both model parameters and internal buffers:", body_style))

    code_txt = "alpha = min(1.0 - 1.0 / (epoch + 1), theta)<br/>for p_tea, p_stu in zip(model_tea.parameters(), model_stu.parameters()):<br/>&nbsp;&nbsp;&nbsp;&nbsp;p_tea.data.mul_(alpha).add_(p_stu.data, alpha=1.0 - alpha)<br/>for b_tea, b_stu in zip(model_tea.buffers(), model_stu.buffers()):<br/>&nbsp;&nbsp;&nbsp;&nbsp;if b_tea.dtype.is_floating_point:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b_tea.data.mul_(alpha).add_(b_stu.data, alpha=1.0 - alpha)<br/>&nbsp;&nbsp;&nbsp;&nbsp;else:<br/>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;b_tea.data.copy_(b_stu.data)"
    story.append(Paragraph(code_txt, ParagraphStyle('CodeBlock', parent=body_style, fontName='Courier-Bold', fontSize=6.8, leading=9.5, textColor=c_primary, backColor=c_light_bg, leftIndent=10, spaceBefore=3, spaceAfter=4)))

    story.append(Paragraph("Pre-flight unit tests verified active tracking across 36/36 Teacher BatchNorm buffers, guaranteeing numerical stability for subsequent experiments.", body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("11.6 EXP-MLUA-003 (Corrected 60-Epoch Benchmark Run)", h2_style))
    story.append(Paragraph("<b>EXP-MLUA-003</b> evaluated the fully remediated architecture (ResNet-34 + FPN, 530 labeled, 1,859 unlabeled patches, seed 42) across 60 complete epochs (7,920 global steps). Training completed flawlessly with zero numerical anomalies. Table 5 provides the complete 60-epoch training history.", body_style))

    fig8_p = os.path.join(out_dir, "figure8_exp003_dice.png")
    if os.path.exists(fig8_p):
        story.append(Image(fig8_p, width=440, height=190))
        story.append(Paragraph("<b>Figure 8:</b> EXP-MLUA-003 validation Dice progression across all 60 training epochs, highlighting peak checkpoint at Epoch 56.", caption_style))

    fig9_p = os.path.join(out_dir, "figure9_exp003_loss.png")
    if os.path.exists(fig9_p):
        story.append(Image(fig9_p, width=440, height=190))
        story.append(Paragraph("<b>Figure 9:</b> EXP-MLUA-003 training loss and validation loss curves across 60 epochs.", caption_style))
    story.append(Spacer(1, 4))

    # TABLE 5: COMPLETE 60-EPOCH HISTORY (Chunked for clear readability)
    story.append(Paragraph("11.7 EXP-MLUA-003 Complete Epoch-by-Epoch History (E1 to E60)", h2_style))
    df3 = pd.read_csv("outputs/experiments/EXP-MLUA-003_FINAL/EXP-MLUA-003_FULL_TRAINING_HISTORY.csv")

    # Table 5A: Epochs 1 to 20
    t5a_data = [[Paragraph("<b>Epoch</b>", table_header_style), Paragraph("<b>Train Loss</b>", table_header_style), Paragraph("<b>Val Loss</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>IoU (%)</b>", table_header_style), Paragraph("<b>Precision (%)</b>", table_header_style), Paragraph("<b>Recall (%)</b>", table_header_style), Paragraph("<b>Max FG</b>", table_header_style), Paragraph("<b>Zero-Pred</b>", table_header_style)]]
    for i in range(20):
        row = df3.iloc[i]
        t5a_data.append([
            Paragraph(f"E{int(row['epoch']):02d}", table_cell_bold),
            Paragraph(f"{row['train_loss']:.4f}", table_cell_style),
            Paragraph(f"{row['val_loss']:.4f}", table_cell_style),
            Paragraph(f"{row['val_dice']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['val_iou']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['val_precision']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['val_recall']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['max_foreground_prob']:.3f}", table_cell_style),
            Paragraph(f"{row['zero_pred_patch_ratio']*100:.1f}%", table_cell_style),
        ])
    t5a = Table(t5a_data, colWidths=[40, 55, 55, 55, 55, 55, 55, 50, 60])
    t5a.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t5a)
    story.append(Paragraph("<b>Table 5A:</b> EXP-MLUA-003 complete training history (Epochs 01 to 20).", caption_style))
    story.append(Spacer(1, 4))

    # Table 5B: Epochs 21 to 40
    t5b_data = [[Paragraph("<b>Epoch</b>", table_header_style), Paragraph("<b>Train Loss</b>", table_header_style), Paragraph("<b>Val Loss</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>IoU (%)</b>", table_header_style), Paragraph("<b>Precision (%)</b>", table_header_style), Paragraph("<b>Recall (%)</b>", table_header_style), Paragraph("<b>Max FG</b>", table_header_style), Paragraph("<b>Zero-Pred</b>", table_header_style)]]
    for i in range(20, 40):
        row = df3.iloc[i]
        t5b_data.append([
            Paragraph(f"E{int(row['epoch']):02d}", table_cell_bold),
            Paragraph(f"{row['train_loss']:.4f}", table_cell_style),
            Paragraph(f"{row['val_loss']:.4f}", table_cell_style),
            Paragraph(f"{row['val_dice']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['val_iou']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['val_precision']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['val_recall']*100:.2f}%", table_cell_style),
            Paragraph(f"{row['max_foreground_prob']:.3f}", table_cell_style),
            Paragraph(f"{row['zero_pred_patch_ratio']*100:.1f}%", table_cell_style),
        ])
    t5b = Table(t5b_data, colWidths=[40, 55, 55, 55, 55, 55, 55, 50, 60])
    t5b.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t5b)
    story.append(Paragraph("<b>Table 5B:</b> EXP-MLUA-003 complete training history (Epochs 21 to 40).", caption_style))
    story.append(Spacer(1, 4))

    # Table 5C: Epochs 41 to 60
    t5c_data = [[Paragraph("<b>Epoch</b>", table_header_style), Paragraph("<b>Train Loss</b>", table_header_style), Paragraph("<b>Val Loss</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>IoU (%)</b>", table_header_style), Paragraph("<b>Precision (%)</b>", table_header_style), Paragraph("<b>Recall (%)</b>", table_header_style), Paragraph("<b>Max FG</b>", table_header_style), Paragraph("<b>Zero-Pred</b>", table_header_style)]]
    for i in range(40, 60):
        row = df3.iloc[i]
        is_best = (int(row['epoch']) == 56)
        st_cell = table_cell_bold if is_best else table_cell_style
        t5c_data.append([
            Paragraph(f"<b>E{int(row['epoch']):02d}{'*' if is_best else ''}</b>", st_cell),
            Paragraph(f"{row['train_loss']:.4f}", st_cell),
            Paragraph(f"{row['val_loss']:.4f}", st_cell),
            Paragraph(f"{row['val_dice']*100:.2f}%", st_cell),
            Paragraph(f"{row['val_iou']*100:.2f}%", st_cell),
            Paragraph(f"{row['val_precision']*100:.2f}%", st_cell),
            Paragraph(f"{row['val_recall']*100:.2f}%", st_cell),
            Paragraph(f"{row['max_foreground_prob']:.3f}", st_cell),
            Paragraph(f"{row['zero_pred_patch_ratio']*100:.1f}%", st_cell),
        ])
    t5c = Table(t5c_data, colWidths=[40, 55, 55, 55, 55, 55, 55, 50, 60])
    t5c.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BACKGROUND', (0,16), (-1,16), c_alert_bg),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 1.8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.8),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t5c)
    story.append(Paragraph("<b>Table 5C:</b> EXP-MLUA-003 complete training history (Epochs 41 to 60, *Epoch 56 selected as peak checkpoint).", caption_style))
    story.append(Spacer(1, 4))

    # SECTION 11.8: FLUCTUATION ANALYSIS
    story.append(Paragraph("11.8 Milestone Progression & Fluctuation Analysis", h2_style))
    story.append(Paragraph("<b>Metric Fluctuation Definitions & Statistics:</b> Fluctuation is quantified as the first-order difference &Delta;Dice<sub>t</sub> = Dice<sub>t</sub> - Dice<sub>t-1</sub> across consecutive epochs:", body_style))
    story.append(Paragraph("&bull; <b>Max Positive Dice Jump:</b> +20.898% at Epoch 21 (from 9.794% at E20 to 30.692% at E21).", bullet_style))
    story.append(Paragraph("&bull; <b>Max Negative Dice Drop:</b> -24.065% at Epoch 22 (from 30.692% at E21 to 6.627% at E22).", bullet_style))
    story.append(Paragraph("&bull; <b>Mean Absolute Epoch-to-Epoch Dice Change:</b> 5.060%.", bullet_style))
    story.append(Paragraph("&bull; <b>Standard Deviation of Dice across Epochs:</b> 23.030%.", bullet_style))
    story.append(Paragraph("&bull; <b>Final vs. Best Difference:</b> -4.096% (E60 Dice = 61.527% vs. E56 Dice = 65.623%).", bullet_style))
    story.append(Paragraph("&bull; <b>Validation Loss Dynamic Range:</b> [0.76388, 1.05110].", bullet_style))
    story.append(Paragraph("&bull; <b>Precision Dynamic Range:</b> [0.000%, 76.100%] (peak precision at Epoch 58).", bullet_style))
    story.append(Paragraph("&bull; <b>Recall Dynamic Range:</b> [0.000%, 63.649%] (peak recall at Epoch 56).", bullet_style))
    story.append(Spacer(1, 4))

    # SECTION 11.9: THRESHOLD SENSITIVITY
    story.append(Paragraph("11.9 Validation Threshold Sensitivity Analysis (&tau; &isin; [0.05, 0.95])", h2_style))
    story.append(Paragraph("Using checkpoint E56, an exhaustive threshold sweep evaluated 19 operating cutoffs from &tau; = 0.05 to 0.95 (step 0.05). Peak validation Dice (65.623%) with optimal precision-recall balance (69.009% vs 63.649%) occurred at <b>&tau; = 0.50</b>, which was frozen as the operating point. Table 6 and Figures 10-11 present these sweep dynamics.", body_style))

    fig10_p = os.path.join(out_dir, "figure10_threshold_dice.png")
    if os.path.exists(fig10_p):
        story.append(Image(fig10_p, width=440, height=190))
        story.append(Paragraph("<b>Figure 10:</b> Checkpoint E56 validation Dice sensitivity across decision thresholds &tau; = 0.05 to 0.95.", caption_style))

    fig11_p = os.path.join(out_dir, "figure11_threshold_prec_rec.png")
    if os.path.exists(fig11_p):
        story.append(Image(fig11_p, width=440, height=190))
        story.append(Paragraph("<b>Figure 11:</b> Checkpoint E56 Precision vs. Recall trade-off curve across decision thresholds.", caption_style))

    t_thresh_data = [
        [Paragraph("<b>Threshold (&tau;)</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>Precision (%)</b>", table_header_style), Paragraph("<b>Recall (%)</b>", table_header_style), Paragraph("<b>Specificity (%)</b>", table_header_style)],
        [Paragraph("&tau; = 0.05", table_cell_style), Paragraph("58.847%", table_cell_style), Paragraph("49.060%", table_cell_style), Paragraph("75.925%", table_cell_style), Paragraph("99.270%", table_cell_style)],
        [Paragraph("&tau; = 0.15", table_cell_style), Paragraph("63.246%", table_cell_style), Paragraph("57.799%", table_cell_style), Paragraph("71.429%", table_cell_style), Paragraph("99.532%", table_cell_style)],
        [Paragraph("&tau; = 0.30", table_cell_style), Paragraph("64.930%", table_cell_style), Paragraph("63.630%", table_cell_style), Paragraph("67.547%", table_cell_style), Paragraph("99.660%", table_cell_style)],
        [Paragraph("<b>&tau; = 0.50*</b>", table_cell_bold), Paragraph("<b>65.623%</b>", table_cell_bold), Paragraph("<b>69.009%</b>", table_cell_bold), Paragraph("<b>63.649%</b>", table_cell_bold), Paragraph("<b>99.753%</b>", table_cell_bold)],
        [Paragraph("&tau; = 0.70", table_cell_style), Paragraph("65.227%", table_cell_style), Paragraph("73.832%", table_cell_style), Paragraph("59.375%", table_cell_style), Paragraph("99.822%", table_cell_style)],
        [Paragraph("&tau; = 0.85", table_cell_style), Paragraph("63.134%", table_cell_style), Paragraph("78.247%", table_cell_style), Paragraph("53.752%", table_cell_style), Paragraph("99.877%", table_cell_style)],
        [Paragraph("&tau; = 0.95", table_cell_style), Paragraph("57.661%", table_cell_style), Paragraph("82.964%", table_cell_style), Paragraph("45.002%", table_cell_style), Paragraph("99.929%", table_cell_style)],
    ]
    t_thresh = Table(t_thresh_data, colWidths=[90, 95, 95, 95, 105])
    t_thresh.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('BACKGROUND', (0,4), (-1,4), c_alert_bg),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_thresh)
    story.append(Paragraph("<b>Table 6:</b> Checkpoint E56 threshold sensitivity analysis (*&tau; = 0.50 selected as operational point).", caption_style))
    story.append(Spacer(1, 4))

    # SECTION 11.10: SEALED TEST RESULTS
    story.append(Paragraph("11.10 Independent Sealed Test Evaluation (100 OPG Cases)", h2_style))
    story.append(Paragraph("The frozen E56 model (&tau; = 0.50) was evaluated on 100 independent sealed panoramic X-rays (117,964,800 evaluated pixels) using sliding-window inference with overlap averaging. The model achieved <b>Macro Dice = 43.041%</b>, <b>Macro IoU = 29.057%</b>, <b>Macro Precision = 41.244%</b>, <b>Macro Recall = 52.896%</b>, and <b>Macro Specificity = 99.630%</b>, with <b>Micro Dice = 43.391%</b> (TP = 263,935, FP = 434,392, FN = 254,282, TN = 117,012,191, Zero-Pred = 0.0%). Table 7 and Figure 12 summarize these benchmarks.", body_style))

    fig12_p = os.path.join(out_dir, "figure12_val_vs_test.png")
    if os.path.exists(fig12_p):
        story.append(Image(fig12_p, width=440, height=190))
        story.append(Paragraph("<b>Figure 12:</b> Validation vs. Independent Sealed Test benchmark metric comparison.", caption_style))

    fig13_p = os.path.join(out_dir, "figure13_visual_overlay.png")
    if os.path.exists(fig13_p):
        story.append(Image(fig13_p, width=480, height=160))
        story.append(Paragraph("<b>Figure 13:</b> Inference pipeline visualization: Input radiograph crop &rarr; Continuous probability map &rarr; Thresholded binary mask (&tau; = 0.50) &rarr; Lesion candidate overlay.", caption_style))

    t_test_data = [
        [Paragraph("<b>Evaluation Metric</b>", table_header_style), Paragraph("<b>Case Macro-Averaged</b>", table_header_style), Paragraph("<b>Global Pixel Micro-Averaged</b>", table_header_style), Paragraph("<b>Confusion Pixel Counts</b>", table_header_style)],
        [Paragraph("Dice Similarity Coefficient", table_cell_bold), Paragraph("<b>43.041%</b>", table_cell_bold), Paragraph("<b>43.391%</b>", table_cell_bold), Paragraph("True Positives (TP): 263,935", table_cell_style)],
        [Paragraph("Intersection-over-Union (IoU)", table_cell_bold), Paragraph("<b>29.057%</b>", table_cell_bold), Paragraph("<b>27.707%</b>", table_cell_bold), Paragraph("False Positives (FP): 434,392", table_cell_style)],
        [Paragraph("Precision (PPV)", table_cell_bold), Paragraph("<b>41.244%</b>", table_cell_bold), Paragraph("<b>37.795%</b>", table_cell_bold), Paragraph("False Negatives (FN): 254,282", table_cell_style)],
        [Paragraph("Recall (Sensitivity)", table_cell_bold), Paragraph("<b>52.896%</b>", table_cell_bold), Paragraph("<b>50.931%</b>", table_cell_bold), Paragraph("True Negatives (TN): 117,012,191", table_cell_style)],
        [Paragraph("Specificity (TNR)", table_cell_bold), Paragraph("<b>99.630%</b>", table_cell_bold), Paragraph("<b>99.630%</b>", table_cell_bold), Paragraph("Total Evaluated: 117,964,800", table_cell_style)],
        [Paragraph("Zero-Prediction Failures", table_cell_bold), Paragraph("<b>0.0% (0 / 100 cases)</b>", table_cell_bold), Paragraph("<b>0.0% (0 / 100 cases)</b>", table_cell_bold), Paragraph("Valid predictions on 100% of scans", table_cell_style)],
    ]
    t_test = Table(t_test_data, colWidths=[130, 115, 115, 120])
    t_test.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_test)
    story.append(Paragraph("<b>Table 7:</b> Independent sealed test results on 100 full panoramic radiographs at &tau; = 0.50.", caption_style))
    story.append(Spacer(1, 4))

    # SECTION 11.11: PER-CASE DISTRIBUTION
    story.append(Paragraph("11.11 Per-Case Sealed Test Distribution & Outlier Analysis", h2_style))
    story.append(Paragraph("Statistical distribution of metrics across all 100 sealed test cases:", body_style))

    t_dist_data = [
        [Paragraph("<b>Metric</b>", table_header_style), Paragraph("<b>Mean</b>", table_header_style), Paragraph("<b>Std Dev</b>", table_header_style), Paragraph("<b>Min</b>", table_header_style), Paragraph("<b>Q25</b>", table_header_style), Paragraph("<b>Median</b>", table_header_style), Paragraph("<b>Q75</b>", table_header_style), Paragraph("<b>Max</b>", table_header_style)],
        [Paragraph("Dice", table_cell_bold), Paragraph("43.04%", table_cell_style), Paragraph("17.87%", table_cell_style), Paragraph("0.998%", table_cell_style), Paragraph("28.77%", table_cell_style), Paragraph("44.11%", table_cell_style), Paragraph("56.80%", table_cell_style), Paragraph("77.94%", table_cell_style)],
        [Paragraph("IoU", table_cell_bold), Paragraph("29.06%", table_cell_style), Paragraph("14.63%", table_cell_style), Paragraph("0.502%", table_cell_style), Paragraph("16.80%", table_cell_style), Paragraph("28.30%", table_cell_style), Paragraph("39.67%", table_cell_style), Paragraph("63.85%", table_cell_style)],
        [Paragraph("Precision", table_cell_bold), Paragraph("41.24%", table_cell_style), Paragraph("21.49%", table_cell_style), Paragraph("0.855%", table_cell_style), Paragraph("24.55%", table_cell_style), Paragraph("39.99%", table_cell_style), Paragraph("55.88%", table_cell_style), Paragraph("91.39%", table_cell_style)],
        [Paragraph("Recall", table_cell_bold), Paragraph("52.90%", table_cell_style), Paragraph("20.42%", table_cell_style), Paragraph("1.200%", table_cell_style), Paragraph("40.90%", table_cell_style), Paragraph("55.15%", table_cell_style), Paragraph("66.94%", table_cell_style), Paragraph("87.07%", table_cell_style)],
        [Paragraph("Specificity", table_cell_bold), Paragraph("99.63%", table_cell_style), Paragraph("0.294%", table_cell_style), Paragraph("98.43%", table_cell_style), Paragraph("99.57%", table_cell_style), Paragraph("99.72%", table_cell_style), Paragraph("99.81%", table_cell_style), Paragraph("99.96%", table_cell_style)],
    ]
    t_dist = Table(t_dist_data, colWidths=[65, 60, 60, 60, 60, 60, 60, 55])
    t_dist.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_dist)
    story.append(Paragraph("<b>Table 8:</b> Statistical distribution of per-case performance across the 100 sealed test cases.", caption_style))

    t_outliers_data = [
        [Paragraph("<b>Top 5 Best Cases</b>", table_header_style), Paragraph("<b>Dice (%)</b>", table_header_style), Paragraph("<b>IoU (%)</b>", table_header_style), Paragraph("<b>Bottom 5 Challenging Cases</b>", table_header_style), Paragraph("<b>Dice (%)</b>", table_header_style), Paragraph("<b>IoU (%)</b>", table_header_style)],
        [Paragraph("Case 1096", table_cell_bold), Paragraph("77.94%", table_cell_style), Paragraph("63.85%", table_cell_style), Paragraph("Case 0392", table_cell_bold), Paragraph("0.998%", table_cell_style), Paragraph("0.502%", table_cell_style)],
        [Paragraph("Case 0748", table_cell_bold), Paragraph("75.25%", table_cell_style), Paragraph("60.32%", table_cell_style), Paragraph("Case 1058", table_cell_bold), Paragraph("2.936%", table_cell_style), Paragraph("1.490%", table_cell_style)],
        [Paragraph("Case 0396", table_cell_bold), Paragraph("73.59%", table_cell_style), Paragraph("58.21%", table_cell_style), Paragraph("Case 0939", table_cell_bold), Paragraph("6.077%", table_cell_style), Paragraph("3.133%", table_cell_style)],
        [Paragraph("Case 0736", table_cell_bold), Paragraph("72.55%", table_cell_style), Paragraph("56.92%", table_cell_style), Paragraph("Case 0925", table_cell_bold), Paragraph("6.157%", table_cell_style), Paragraph("3.176%", table_cell_style)],
        [Paragraph("Case 0954", table_cell_bold), Paragraph("71.55%", table_cell_style), Paragraph("55.70%", table_cell_style), Paragraph("Case 1091", table_cell_bold), Paragraph("14.135%", table_cell_style), Paragraph("7.605%", table_cell_style)],
    ]
    t_outliers = Table(t_outliers_data, colWidths=[90, 70, 70, 110, 70, 70])
    t_outliers.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_outliers)
    story.append(Paragraph("<b>Table 9:</b> Top 5 and bottom 5 performing sealed test cases.", caption_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 12 & 13: MASTER COMPARISON & GENERALIZATION GAP
    # ==========================================
    story.append(Paragraph("12. COMPREHENSIVE EXPERIMENTAL COMPARISON", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))

    t_master_data = [
        [Paragraph("<b>Experiment ID</b>", table_header_style), Paragraph("<b>Method / Architecture</b>", table_header_style), Paragraph("<b>Labeled / Total</b>", table_header_style), Paragraph("<b>Validation Dice</b>", table_header_style), Paragraph("<b>Sealed Test Dice</b>", table_header_style), Paragraph("<b>Experimental Status</b>", table_header_style)],
        [Paragraph("EXP001", table_cell_bold), Paragraph("Supervised DeepLabV3+ (ResNet-18)", table_cell_style), Paragraph("100% Labeled", table_cell_style), Paragraph("36.12%", table_cell_style), Paragraph("Not evaluated", table_cell_style), Paragraph("Completed / Historical", table_cell_style)],
        [Paragraph("EXP002", table_cell_bold), Paragraph("Supervised DeepLabV3+ (Focal)", table_cell_style), Paragraph("100% Labeled", table_cell_style), Paragraph("41.20%", table_cell_style), Paragraph("Not evaluated", table_cell_style), Paragraph("Completed / Historical", table_cell_style)],
        [Paragraph("EXP003", table_cell_bold), Paragraph("Supervised FPN (ResNet-18)", table_cell_style), Paragraph("100% Labeled", table_cell_style), Paragraph("43.85%", table_cell_style), Paragraph("Not evaluated", table_cell_style), Paragraph("Completed / Historical", table_cell_style)],
        [Paragraph("EXP006", table_cell_bold), Paragraph("Supervised DeepLabV3+ (512x512)", table_cell_style), Paragraph("100% Labeled", table_cell_style), Paragraph("48.39%", table_cell_style), Paragraph("Not evaluated", table_cell_style), Paragraph("Peak Supervised Baseline", table_cell_style)],
        [Paragraph("EXP-MLUA-001", table_cell_bold), Paragraph("SSL MLUA ResNet-34+FPN", table_cell_style), Paragraph("10% Labeled", table_cell_style), Paragraph("14.71% (collapsed)", table_cell_style), Paragraph("Not evaluated", table_cell_style), Paragraph("Foreground degradation", table_cell_style)],
        [Paragraph("EXP-MLUA-002", table_cell_bold), Paragraph("SSL MLUA ResNet-34+FPN", table_cell_style), Paragraph("20% Labeled", table_cell_style), Paragraph("0.0% (crashed E10)", table_cell_style), Paragraph("Not evaluated", table_cell_style), Paragraph("NaN crash (unsynced BN)", table_cell_style)],
        [Paragraph("<b>EXP-MLUA-003*</b>", table_cell_bold), Paragraph("<b>SSL MLUA (Dual Buffer Sync)</b>", table_cell_bold), Paragraph("<b>20% Labeled</b>", table_cell_bold), Paragraph("<b>65.62% (E56)</b>", table_cell_bold), Paragraph("<b>43.04% (Macro)</b>", table_cell_bold), Paragraph("<b>Selected Final Model</b>", table_cell_bold)],
    ]
    t_master = Table(t_master_data, colWidths=[75, 125, 70, 75, 65, 70])
    t_master.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BACKGROUND', (0,7), (-1,7), c_alert_bg),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
    ]))
    story.append(t_master)
    story.append(Paragraph("<b>Table 10:</b> Master experimental comparison across supervised baselines and MLUA semi-supervised trials.", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("13. VALIDATION VS. SEALED TEST GENERALIZATION GAP", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    story.append(Paragraph("A significant performance gap exists between patch-level validation Dice (65.623%) and full-OPG sealed test Macro Dice (43.041%, &Delta; = -22.582 pp). Forensic error analysis indicates:", body_style))
    story.append(Paragraph("&bull; <b>Precision Degradation (&Delta; = -27.765 pp):</b> Full panoramic reconstruction exposes the model to background anatomical structures (cervical burnout, metal restorations, maxillary sinus) that generate 434,392 false-positive pixels.", bullet_style))
    story.append(Paragraph("&bull; <b>Recall Retention (&Delta; = -10.753 pp):</b> The model maintained 52.896% recall on full OPGs, successfully capturing true proximal and occlusal cavities.", bullet_style))
    story.append(Paragraph("&bull; <b>Specificity Stability (&Delta; = -0.123 pp):</b> Specificity remained near-perfect (99.630%), confirming robust rejection of healthy jawbone and background air space.", bullet_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 14 & 15: CURRENT STATUS & REMAINING WORK
    # ==========================================
    story.append(Paragraph("14. CURRENT STATUS SUMMARY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))

    t_status_data = [
        [Paragraph("<b>Component / Milestone</b>", table_header_style), Paragraph("<b>Implementation Status</b>", table_header_style), Paragraph("<b>Experimental Evidence / Location</b>", table_header_style)],
        [Paragraph("DC1000 Dataset Ingestion & Patching", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("2,389 training patches standardized to 384x384.", table_cell_style)],
        [Paragraph("Supervised Baselines (EXP001-006)", table_cell_bold), Paragraph("Completed / Historical", table_cell_style), Paragraph("DeepLabV3+ achieved peak Dice 48.39% at 512x512.", table_cell_style)],
        [Paragraph("EXP-MLUA-001 & 002 Trials", table_cell_bold), Paragraph("Completed / Historical", table_cell_style), Paragraph("Documented degradation and NaN crash at Step 1257.", table_cell_style)],
        [Paragraph("Forensic Buffer Sync Remediation", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("Dual parameter + BN buffer EMA sync implemented.", table_cell_style)],
        [Paragraph("EXP-MLUA-003 Benchmark (60 Epochs)", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("Flawless run; E56 selected (Val Dice 65.62%).", table_cell_style)],
        [Paragraph("Threshold Sensitivity Sweep", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("tau = 0.50 selected on validation cohort.", table_cell_style)],
        [Paragraph("Independent Sealed Test (100 OPGs)", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("Macro Dice = 43.041%, Micro Dice = 43.391%.", table_cell_style)],
        [Paragraph("False-Positive Reduction & App Polish", table_cell_bold), Paragraph("In Progress", table_cell_style), Paragraph("Active focus for next capstone sprint.", table_cell_style)],
    ]
    t_status = Table(t_status_data, colWidths=[140, 100, 240])
    t_status.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t_status)
    story.append(Paragraph("<b>Table 11:</b> Current implementation and experimental status across capstone milestones.", caption_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("15. WORK REMAINING & CAPSTONE DELIVERABLES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))

    t_rem_data = [
        [Paragraph("<b>Remaining Deliverable</b>", table_header_style), Paragraph("<b>Detailed Technical Scope</b>", table_header_style), Paragraph("<b>Target Objective / Metric</b>", table_header_style)],
        [Paragraph("1. False-Positive Filtering", table_cell_bold), Paragraph("Implement morphological and anatomical prior filters to suppress cervical burnout false alarms.", table_cell_style), Paragraph("Improve sealed test Precision from 41.2% to >50%.", table_cell_style)],
        [Paragraph("2. Test Error Taxonomy", table_cell_bold), Paragraph("Perform systematic categorization of failure modes on bottom 10% test cases (diffuse vs. overlap).", table_cell_style), Paragraph("Publish error taxonomy and failure analysis.", table_cell_style)],
        [Paragraph("3. Full Interface Integration", table_cell_bold), Paragraph("Finalize integration between frozen E56 inference backend and interactive web UI demo.", table_cell_style), Paragraph("Enable real-time drag-and-drop OPG analysis.", table_cell_style)],
        [Paragraph("4. Capstone Documentation", table_cell_bold), Paragraph("Complete final project thesis, presentation slides, and open-source codebase packaging.", table_cell_style), Paragraph("Ready for final capstone defense.", table_cell_style)],
    ]
    t_rem = Table(t_rem_data, colWidths=[120, 210, 150])
    t_rem.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t_rem)
    story.append(Paragraph("<b>Table 12:</b> Remaining capstone deliverables and target technical objectives.", caption_style))
    story.append(Spacer(1, 6))

    # ==========================================
    # SECTION 16, 17, 18: CONCLUSION, FUTURE SCOPE, REFERENCES
    # ==========================================
    story.append(Paragraph("16. CONCLUSION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    story.append(Paragraph("This capstone project progressed from studying the published MLUA framework through multiple supervised experiments (EXP001-EXP006), initial semi-supervised trials, diagnostic forensic analysis of numerical instability in EXP-MLUA-002, implementation remediation of Teacher BatchNorm buffer synchronization, and benchmark 60-epoch validation in <b>EXP-MLUA-003</b>. The selected <b>Epoch 56 checkpoint</b> achieved <b>65.623% Dice</b> on validation data and <b>43.041% Macro Dice</b> on 100 independent sealed test cases. These results demonstrate significant development progress while establishing clear technical priorities for our remaining false-positive reduction and application finalization tasks.", body_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("17. FUTURE RESEARCH SCOPE", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    fs_list = [
        "1. <b>Anatomical Prior Masking:</b> Integrate dentition boundary detectors to eliminate background mandibular false positives.",
        "2. <b>Boundary-Aware Losses:</b> Explore Boundary IoU or Hausdorff distance losses to refine demineralization edge localization.",
        "3. <b>Multi-Center Generalization:</b> Validate frozen checkpoints across multi-institutional radiographic archives.",
        "4. <b>Transformer Backbones:</b> Investigate hybrid CNN-Transformer architectures (e.g., Swin Transformer) for long-range context.",
        "5. <b>Clinical Translation:</b> Transition from research benchmarks toward prospective clinical reader studies."
    ]
    for fs in fs_list:
        story.append(Paragraph(fs, bullet_style))
    story.append(Spacer(1, 4))

    story.append(Paragraph("18. REFERENCES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=6))
    refs = [
        "[1] X. Wang, S. Gao, K. Jiang, H. Zhang, L. Wang, F. Chen, J. Yu, and F. Yang, \"Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation,\" <i>Neurocomputing</i>, vol. 540, p. 126208, 2023. DOI: 10.1016/j.neucom.2023.03.069.",
        "[2] X. Wang, S. Gao, et al., \"Official MLUA Research Codebase,\" GitHub Repository: https://github.com/Zzz512/MLUA, 2023.",
        "[3] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" in <i>Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)</i>, 2016, pp. 770-778.",
        "[4] T.-Y. Lin, P. Doll&aacute;r, R. Girshick, K. He, B. Hariharan, and S. Belongie, \"Feature pyramid networks for object detection,\" in <i>Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)</i>, 2017, pp. 2117-2125.",
        "[5] A. Tarvainen and H. Valpola, \"Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017.",
        "[6] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,\" in <i>International Conference on Machine Learning (ICML)</i>, 2016, pp. 1050-1059.",
        "[7] L.-C. Chen, G. Papandreou, F. Schroff, and H. Adam, \"Rethinking atrous convolution for semantic image segmentation,\" in <i>arXiv:1706.05587</i>, 2017.",
        "[8] D. Jha et al., \"DoubleU-Net: A deep convolutional neural network for medical image segmentation,\" in <i>Proc. IEEE CBMS</i>, 2020, pp. 558-564."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefP', parent=body_style, leftIndent=16, firstLineIndent=-12, spaceAfter=2.5)))

    doc.build(story, canvasmaker=DetailedProgressNumberedCanvas)
    print(f"Successfully compiled Detailed Capstone Progress Report: {output_filename}")

if __name__ == "__main__":
    create_detailed_progress_pdf()
