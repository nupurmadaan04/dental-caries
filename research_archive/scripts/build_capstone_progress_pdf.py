import os
import sys
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak, HRFlowable
)
from reportlab.pdfgen import canvas

class ProgressNumberedCanvas(canvas.Canvas):
    """
    Two-pass canvas for dynamic total page numbering,
    running header and running footer on capstone report.
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
        header_text = "B.Tech Capstone Project Progress Report | Dental Caries Segmentation (MLUA)"
        self.drawString(54, A4[1] - 36, header_text)
        self.setStrokeColor(colors.HexColor("#CBD5E0"))
        self.setLineWidth(0.6)
        self.line(54, A4[1] - 42, A4[0] - 54, A4[1] - 42)

        # Running Footer
        footer_left = "Student: Nupur Madaan (B.Tech CSE, 2026-27)"
        footer_right = f"Page {self._pageNumber} of {page_count}"
        self.drawString(54, 34, footer_left)
        self.drawRightString(A4[0] - 54, 34, footer_right)
        self.line(54, 46, A4[0] - 54, 46)

        self.restoreState()


def create_progress_pdf(output_filename="Dental_Caries_Segmentation_Capstone_Progress_Report.pdf"):
    print(f"Compiling Capstone Progress Report: {output_filename}...")
    
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=A4,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()
    
    # Custom Academic Palette
    c_primary = colors.HexColor("#1A365D")   # Deep Navy
    c_secondary = colors.HexColor("#2B6CB0") # Slate Blue
    c_dark = colors.HexColor("#2D3748")      # Charcoal Body Text
    c_light_bg = colors.HexColor("#F7FAFC")  # Panel background
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
        spaceAfter=20
    )

    h1_style = ParagraphStyle(
        'ProgH1',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_primary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'ProgH2',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=10,
        leading=13.5,
        textColor=c_secondary,
        spaceBefore=8,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'ProgBody',
        parent=styles['BodyText'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=c_dark,
        spaceAfter=5,
        alignment=4 # Justified
    )

    bullet_style = ParagraphStyle(
        'ProgBullet',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=3,
        alignment=4
    )

    caption_style = ParagraphStyle(
        'ProgCaption',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor("#4A5568"),
        alignment=1,
        spaceBefore=3,
        spaceAfter=8
    )

    table_header_style = ParagraphStyle(
        'ProgTH',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.5,
        leading=9.5,
        textColor=colors.white,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'ProgTD',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=7.2,
        leading=9.5,
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
    # 1. COVER PAGE
    # ==========================================
    story.append(Spacer(1, 20))
    story.append(Paragraph("B.TECH CAPSTONE PROJECT PROGRESS REPORT", ParagraphStyle('TopTag', fontName='Helvetica-Bold', fontSize=10, textColor=c_secondary, alignment=1, spaceAfter=10)))
    story.append(HRFlowable(width="100%", thickness=2, color=c_primary, spaceAfter=20))
    
    story.append(Paragraph("Progress Report on Dental Caries Segmentation from Panoramic X-Ray Images Using Multi-Level Uncertainty-Aware Learning", title_style))
    story.append(Paragraph("A Study on Baseline Implementation, Training Diagnostics, Buffer Remediation, and Empirical Validation on the DC1000 Benchmark", subtitle_style))
    
    story.append(Spacer(1, 25))

    meta_table = [
        [Paragraph("<b>Student Name:</b>", table_cell_bold), Paragraph("Nupur Madaan", table_cell_style)],
        [Paragraph("<b>Degree Program:</b>", table_cell_bold), Paragraph("Bachelor of Technology (B.Tech) in Computer Science & Engineering", table_cell_style)],
        [Paragraph("<b>Enrollment / Roll No.:</b>", table_cell_bold), Paragraph("[Placeholder: Student Enrollment Number]", table_cell_style)],
        [Paragraph("<b>Department:</b>", table_cell_bold), Paragraph("[Placeholder: Department of Computer Science & Engineering]", table_cell_style)],
        [Paragraph("<b>Institution / University:</b>", table_cell_bold), Paragraph("[Placeholder: University / Institute Name]", table_cell_style)],
        [Paragraph("<b>Faculty Guide / Supervisor:</b>", table_cell_bold), Paragraph("[Placeholder: Faculty Supervisor / Guide Name]", table_cell_style)],
        [Paragraph("<b>Project Domain:</b>", table_cell_bold), Paragraph("Medical Computer Vision & Deep Learning (Dental Imaging)", table_cell_style)],
        [Paragraph("<b>Current Major Milestone:</b>", table_cell_bold), Paragraph("EXP-MLUA-003 (60-Epoch Remediated Run & Sealed Evaluation Completed)", table_cell_style)],
        [Paragraph("<b>Academic Year:</b>", table_cell_bold), Paragraph("2026-27", table_cell_style)],
    ]
    t_meta = Table(meta_table, colWidths=[150, 310])
    t_meta.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_light_bg),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4.5),
        ('TOPPADDING', (0,0), (-1,-1), 4.5),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('LINEBELOW', (0,0), (-1,-2), 0.5, c_border),
    ]))
    story.append(t_meta)
    
    story.append(Spacer(1, 30))

    notice_box = [
        [Paragraph("<b>Progress Narrative Summary:</b> This progress report documents the progressive engineering and experimental journey of our capstone project. Starting from the published baseline paper (Wang et al., <i>Neurocomputing</i> 2023), we implemented the semi-supervised Teacher-Student framework, investigated early training degradation (EXP-001) and numerical collapse (EXP-002), isolated the root cause to Teacher BatchNorm buffer desynchronization, remediated the update pipeline in EXP-003, validated the resulting model (Epoch 56 Val Dice = 65.62%), completed independent sealed testing (Macro Dice = 43.04%), and identified the specific remaining tasks required for capstone completion.", ParagraphStyle('NoticeP', parent=body_style, fontSize=7.8, leading=11))]
    ]
    t_notice = Table(notice_box, colWidths=[480])
    t_notice.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), c_alert_bg),
        ('BOX', (0,0), (-1,-1), 1, c_border),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_notice)
    story.append(PageBreak())

    # ==========================================
    # 2. TABLE OF CONTENTS
    # ==========================================
    story.append(Paragraph("TABLE OF CONTENTS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=10))

    toc_rows = [
        ("1. ABSTRACT", "3"),
        ("2. INTRODUCTION", "4"),
        ("3. OBJECTIVES", "5"),
        ("4. PROBLEM STATEMENT", "5"),
        ("5. PROJECT PROGRESS / WORK COMPLETED", "6"),
        ("6. FLOWCHART & SYSTEM ARCHITECTURE", "8"),
        ("7. METHODOLOGY", "10"),
        ("    7.1 Input Collection & DC1000 Benchmark", "10"),
        ("    7.2 Dataset Characteristics & Binary Segmentation Scope", "10"),
        ("    7.3 Data Preparation & 20/80 Partitioning", "10"),
        ("    7.4 Preprocessing & Normalization", "11"),
        ("    7.5 384x384 Patch Extraction & Sliding Windows", "11"),
        ("    7.6 ResNet-34 Feature Extraction Backbone", "11"),
        ("    7.7 Feature Pyramid Network (FPN) Decoder", "11"),
        ("    7.8 Multi-Scale Auxiliary & Fused Segmentation Heads", "12"),
        ("    7.9 Teacher-Student Consistency Learning", "12"),
        ("    7.10 Loss Functions (BCE + Soft Dice + Consistency)", "12"),
        ("    7.11 Teacher EMA Update & Buffer Remediation", "12"),
        ("    7.12 Numerical Instability Analysis (EXP-MLUA-002)", "13"),
        ("    7.13 Monte Carlo Uncertainty Estimation & Confidence Masking", "13"),
        ("    7.14 Threshold Selection (Validation Sweep)", "13"),
        ("    7.15 Sliding-Window Panoramic Reconstruction", "14"),
        ("    7.16 Final Output Representation", "14"),
        ("8. SYSTEM IMPLEMENTATION MODULES", "14"),
        ("9. EXPERIMENTAL PROGRESS AND RESULTS", "16"),
        ("    9.1 Published MLUA Baseline Context", "16"),
        ("    9.2 EXP-MLUA-001 (Baseline Run & Degradation)", "16"),
        ("    9.3 EXP-MLUA-002 (Numerical Instability & Collapse)", "16"),
        ("    9.4 Forensic Diagnostic Analysis of EXP-MLUA-002", "17"),
        ("    9.5 Remediation: Dual Parameter & Buffer EMA Synchronization", "17"),
        ("    9.6 EXP-MLUA-003 (Corrected 60-Epoch Run)", "17"),
        ("    9.7 Best Checkpoint Selection (Epoch 56)", "18"),
        ("    9.8 Validation Threshold Sensitivity Analysis", "18"),
        ("    9.9 Independent Sealed Test Evaluation (100 OPGs)", "19"),
        ("    9.10 Validation vs. Sealed Test Generalization Gap", "19"),
        ("10. CURRENT STATUS AND WORK REMAINING", "20"),
        ("11. CONCLUSION", "21"),
        ("12. FUTURE SCOPE", "21"),
        ("13. REFERENCES", "22")
    ]

    toc_table_data = []
    for item, pg in toc_rows:
        is_h = not item.startswith("    ")
        st = table_cell_bold if is_h else table_cell_style
        toc_table_data.append([Paragraph(item, st), Paragraph(f"Page {pg}", table_cell_center)])

    t_toc = Table(toc_table_data, colWidths=[400, 80])
    t_toc.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 1.5),
        ('TOPPADDING', (0,0), (-1,-1), 1.5),
        ('LINEBELOW', (0,0), (-1,-1), 0.3, colors.HexColor("#EDF2F7")),
    ]))
    story.append(t_toc)
    story.append(PageBreak())

    # ==========================================
    # SECTION 1: ABSTRACT
    # ==========================================
    story.append(Paragraph("1. ABSTRACT", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))
    
    p_abs1 = "Automated localization and segmentation of dental caries in orthopantomograms (panoramic dental X-rays) is a critical task in digital oral healthcare. However, developing reliable deep learning segmentation models for panoramic radiographs is severely hindered by extreme class imbalance (lesions occupy <0.5% of total image area), low local radiographic contrast, diffuse demineralization boundaries, anatomical superimpositions, and the prohibitive cost of expert manual annotations. To address clinical annotation scarcity, this capstone project investigates semi-supervised learning based on Multi-Level Uncertainty Aware (MLUA) Teacher-Student consistency architectures using the public DC1000 dataset (1,000 panoramic radiographs)."
    story.append(Paragraph(p_abs1, body_style))

    p_abs2 = "This progress report outlines our complete development journey, from studying the original MLUA publication (Wang et al., <i>Neurocomputing</i> 2023) to conducting sequential experimental trials. In our initial experiments, the baseline model exhibited training degradation and severe foreground suppression (EXP-MLUA-001). Subsequent scaling in EXP-MLUA-002 encountered catastrophic numerical instability (NaN loss collapse during Teacher forward passes). A systematic forensic audit isolated the root cause: during Exponential Moving Average (EMA) updates, Teacher model weights were updated while internal BatchNorm running statistics remained un-synchronized at default values, leading to explosive GroupNorm activations. We resolved this issue by implementing dual parameter and buffer EMA synchronization in <b>EXP-MLUA-003</b>, which successfully completed 60 epochs without instability."
    story.append(Paragraph(p_abs2, body_style))

    p_abs3 = "The remediated model achieved a peak <b>Validation Dice of 65.623%</b>, <b>IoU of 49.854%</b>, <b>Precision of 69.009%</b>, and <b>Recall of 63.649%</b> at <b>Epoch 56</b> with an operating threshold of &tau; = 0.50 on 384&times;384 patches. When evaluated on an independent, frozen sealed test set of 100 full panoramic images, the model achieved a <b>Macro Dice of 43.041%</b>, <b>Macro Recall of 52.896%</b>, and <b>Specificity of 99.630%</b> (Micro Dice = 43.391%). The observed generalization gap highlights key remaining challenges, including background false-positive reduction on cervical burnout artifacts and boundary calibration. Current work is focused on error analysis, post-processing refinement, and final application interface integration."
    story.append(Paragraph(p_abs3, body_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 2: INTRODUCTION
    # ==========================================
    story.append(Paragraph("2. INTRODUCTION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    story.append(Paragraph("Dental caries is among the most widespread chronic oral diseases globally. Left untreated, bacterial acid demineralization penetrates the enamel-dentin junction into the dental pulp, causing severe pain, periapical infection, and tooth loss. Orthopantomography (panoramic dental radiography) is the most widely utilized screening modality in clinical dentistry because a single scan captures the entire maxillofacial region, including all maxillary and mandibular teeth, supporting alveolar bone, and adjacent anatomical structures.", body_style))

    story.append(Paragraph("Despite its clinical ubiquity, automated caries detection on panoramic X-rays presents substantial computer vision challenges:", body_style))
    story.append(Paragraph("&bull; <b>Extreme Class Imbalance:</b> Carious demineralization regions occupy minute spatial areas relative to the full panoramic canvas (often less than 0.5% of total pixels).", bullet_style))
    story.append(Paragraph("&bull; <b>Ambiguous Attenuation & Diffuse Margins:</b> Incipient caries appears as subtle radiolucency with gradual, ill-defined boundaries rather than sharp edges.", bullet_style))
    story.append(Paragraph("&bull; <b>Confounding Anatomical Artifacts:</b> Normal radiographic phenomena, such as cervical burnout at the tooth neck, overlapping proximal tooth contours, and ghost images from contralateral bones, closely mimic true carious lesions.", bullet_style))
    story.append(Paragraph("&bull; <b>Annotation Scarcity:</b> Creating dense, pixel-level ground truth contours requires hours of labor by expert dental clinicians, making large fully supervised datasets difficult to obtain.", bullet_style))

    story.append(Paragraph("To address these challenges, our project explores semi-supervised deep learning inspired by the Multi-Level Uncertainty Aware (MLUA) framework. Semi-supervised learning utilizes a small fraction of labeled images alongside a larger collection of unlabeled images. By employing a Teacher-Student consistency model with Monte Carlo uncertainty estimation, the framework generates pseudo-labels for unlabeled regions while dynamically filtering out unreliable predictions.", body_style))

    story.append(Paragraph("This report presents our project's progressive milestones, detailing how initial experimental failures were analyzed and systematically resolved to establish a functional, validated segmentation pipeline.", body_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 3 & 4: OBJECTIVES & PROBLEM STATEMENT
    # ==========================================
    story.append(Paragraph("3. OBJECTIVES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))
    
    objs = [
        "1. Study the foundational MLUA semi-supervised learning framework proposed for dental panoramic caries segmentation.",
        "2. Implement the ResNet-34 encoder and Feature Pyramid Network (FPN) decoder architecture on the benchmark DC1000 dataset.",
        "3. Investigate model behavior and training dynamics across progressive experimental iterations (EXP-MLUA-001, 002, and 003).",
        "4. Diagnose and resolve the numerical instability (NaN loss) encountered in Teacher forward passes during early training.",
        "5. Correct the Teacher Exponential Moving Average (EMA) update routine by incorporating strict BatchNorm buffer synchronization.",
        "6. Evaluate the corrected model over 60 epochs to select the optimal validation checkpoint (Epoch 56).",
        "7. Conduct a validation threshold sensitivity sweep (&tau; &isin; [0.05, 0.95]) to establish the optimal operating cutoff (&tau; = 0.50).",
        "8. Perform independent evaluation on a frozen sealed test set of 100 panoramic images to assess real-world generalization.",
        "9. Document the observed generalization gap and define the remaining development tasks required for capstone completion."
    ]
    for o in objs:
        story.append(Paragraph(o, bullet_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("4. PROBLEM STATEMENT", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))
    
    ps_text = "<b>Formal Problem Definition:</b> Given a digitized panoramic dental X-ray image <i>X</i> &isin; &real;<sup>H &times; W</sup>, the objective is to develop a deep learning system that outputs a pixel-level probability map <i>P(Y = 1 | X)</i> representing suspected carious demineralization, and converts it into a binary segmentation mask <i>&Ycirc;<sub>&tau;</sub></i> &isin; {0, 1}<sup>H &times; W</sup> at threshold &tau; = 0.50. The model must maximize overlap with expert consensus annotations while maintaining robust numerical stability and operating under limited labeled training data (20% labeled, 80% unlabeled)."
    story.append(Paragraph(ps_text, body_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 5: PROJECT PROGRESS / WORK COMPLETED
    # ==========================================
    story.append(Paragraph("5. PROJECT PROGRESS / WORK COMPLETED", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    story.append(Paragraph("Rather than representing an isolated experiment, this capstone project represents a structured engineering progression. Table 1 summarizes the key phases completed to date.", body_style))

    t1_data = [
        [Paragraph("<b>Development Phase</b>", table_header_style), Paragraph("<b>Work Performed</b>", table_header_style), Paragraph("<b>Key Result / Observation</b>", table_header_style), Paragraph("<b>Status</b>", table_header_style)],
        [Paragraph("1. Literature Study", table_cell_bold), Paragraph("Analyzed Wang et al. (2023) MLUA paper and official repository.", table_cell_style), Paragraph("Understood Teacher-Student SSL and MC uncertainty principles.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("2. Data Preparation", table_cell_bold), Paragraph("Standardized DC1000 images (768&times;1536) & extracted 384&times;384 patches.", table_cell_style), Paragraph("Prepared 530 labeled and 1,859 unlabeled patches (20/80 split).", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("3. Baseline Implementation", table_cell_bold), Paragraph("Constructed PyTorch ResNet-34 + FPN pipeline with auxiliary heads.", table_cell_style), Paragraph("Verified data loading, forward passes, and basic loss modules.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("4. EXP-MLUA-001", table_cell_bold), Paragraph("Conducted initial baseline training run on DC1000 patches.", table_cell_style), Paragraph("Observed severe foreground suppression and sub-optimal Dice.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("5. EXP-MLUA-002", table_cell_bold), Paragraph("Scaled consistency weighting and loss hyper-parameters.", table_cell_style), Paragraph("Encountered fatal NaN loss collapse in Teacher GroupNorm.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("6. Forensic Audit", table_cell_bold), Paragraph("Audited intermediate activations, layer stats, and buffer states.", table_cell_style), Paragraph("Isolated missing Teacher BatchNorm buffer synchronization during EMA.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("7. Buffer Remediation", table_cell_bold), Paragraph("Implemented dual parameter & buffer EMA update in training loop.", table_cell_style), Paragraph("Pre-flight testing confirmed stable, bounded feature activations.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("8. EXP-MLUA-003", table_cell_bold), Paragraph("Executed 60-epoch remediated training run (7,920 global steps).", table_cell_style), Paragraph("Training completed smoothly; peak Val Dice = 65.62% at Epoch 56.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("9. Threshold Analysis", table_cell_bold), Paragraph("Evaluated validation threshold sweep from &tau; = 0.05 to 0.95.", table_cell_style), Paragraph("&tau; = 0.50 identified as optimal validation operating cutoff.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("10. Sealed Testing", table_cell_bold), Paragraph("Evaluated frozen E56 model on 100 independent sealed OPGs.", table_cell_style), Paragraph("Achieved Macro Dice = 43.04%, Micro Dice = 43.39%, Spec = 99.63%.", table_cell_style), Paragraph("Completed", table_cell_center)],
        [Paragraph("11. Application Integration", table_cell_bold), Paragraph("Designed inference pipeline and sliding-window reconstruction.", table_cell_style), Paragraph("Backend reconstruction verified; UI polishing underway.", table_cell_style), Paragraph("In Progress", table_cell_center)],
    ]
    t1 = Table(t1_data, colWidths=[90, 140, 180, 70])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t1)
    story.append(Paragraph("<b>Table 1:</b> Chronological progression of project milestones, experimental findings, and implementation status.", caption_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 6: FLOWCHART
    # ==========================================
    story.append(Paragraph("6. FLOWCHART & SYSTEM ARCHITECTURE", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    story.append(Paragraph("The structured development journey from initial paper review to final evaluation is depicted in Figure 1, while the complete technical system pipeline is shown in Figure 2.", body_style))

    fig1_p = os.path.join("outputs", "progress_figures", "fig1_progress_flowchart.png")
    if os.path.exists(fig1_p):
        story.append(Image(fig1_p, width=480, height=240))
        story.append(Paragraph("<b>Figure 1:</b> Project development and experimentation flowchart illustrating the step-by-step progress from literature study through forensic remediation to final sealed evaluation.", caption_style))

    fig2_p = os.path.join("outputs", "progress_figures", "fig2_system_flowchart.png")
    if os.path.exists(fig2_p):
        story.append(Image(fig2_p, width=480, height=270))
        story.append(Paragraph("<b>Figure 2:</b> Overall system architecture and semi-supervised data flow for EXP-MLUA-003, showing patch extraction, Teacher-Student consistency, confidence masking, and sliding-window reconstruction.", caption_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 7: METHODOLOGY
    # ==========================================
    story.append(Paragraph("7. METHODOLOGY", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    story.append(Paragraph("7.1 Input Collection & DC1000 Benchmark", h2_style))
    story.append(Paragraph("The project uses the publicly available <b>DC1000</b> dental caries dataset, containing 1,000 digitized panoramic radiographs acquired in clinical settings. The dataset provides expert annotations for over 7,500 individual carious lesions, categorized into detailed (593 cases) and rough (407 cases) annotation subsets.", body_style))

    story.append(Paragraph("7.2 Dataset Characteristics & Binary Segmentation Scope", h2_style))
    story.append(Paragraph("While the original dataset documentation mentions clinical caries depth levels (shallow, middle, deep), our current system is designed strictly for <b>pixel-level binary semantic segmentation</b> (0 = Background / Healthy, 1 = Suspected Dental Caries). This focus ensures maximum spatial localization accuracy without imposing multi-class boundary ambiguities.", body_style))

    story.append(Paragraph("7.3 Data Preparation & 20/80 Partitioning", h2_style))
    story.append(Paragraph("To simulate realistic clinical annotation constraints, the training dataset was partitioned into <b>20% labeled data (530 patches)</b> and <b>80% unlabeled data (1,859 patches)</b>, totaling 2,389 training patches. An independent validation set of 598 patches and a sealed test set of 100 full panoramic images were preserved with strict patient-level separation.", body_style))

    story.append(Paragraph("7.4 Preprocessing & Normalization", h2_style))
    story.append(Paragraph("Raw radiographs with varying dimensions were standardized to 768&times;1536 pixels. Pixel intensities were scaled to [0, 1] and normalized using standard ImageNet mean and variance constants (&mu; = [0.485, 0.456, 0.406], &sigma; = [0.229, 0.224, 0.225]).", body_style))

    story.append(Paragraph("7.5 384&times;384 Patch Extraction & Sliding Windows", h2_style))
    story.append(Paragraph("To preserve fine lesion textures without exceeding GPU memory capacity, panoramic images were divided into <b>384&times;384 pixel patches</b> using a sliding window with vertical and horizontal stride S = 192 (50% overlap), yielding 21 patches per panoramic image.", body_style))

    story.append(Paragraph("7.6 ResNet-34 Feature Extraction Backbone", h2_style))
    story.append(Paragraph("The encoder uses a deep ResNet-34 architecture pre-trained on ImageNet. Its residual connections facilitate gradient propagation, generating multi-scale feature hierarchies across stages C<sub>2</sub> (64 ch), C<sub>3</sub> (128 ch), C<sub>4</sub> (256 ch), and C<sub>5</sub> (512 ch).", body_style))

    story.append(Paragraph("7.7 Feature Pyramid Network (FPN) Decoder", h2_style))
    story.append(Paragraph("The FPN decoder combines deep semantic features with shallow spatial details. Lateral 1&times;1 convolutions unify channel dimensions to 256, and top-down pathways upsample and merge features to create pyramid levels P<sub>2</sub>, P<sub>3</sub>, P<sub>4</sub>, and P<sub>5</sub>.", body_style))

    story.append(Paragraph("7.8 Multi-Scale Auxiliary & Fused Segmentation Heads", h2_style))
    story.append(Paragraph("Four auxiliary convolution heads output intermediate segmentation logits from P<sub>2</sub>-P<sub>5</sub>, providing deep supervision during training. A primary fused head concatenates and upsamples all pyramid levels to generate the final 384&times;384 prediction.", body_style))

    story.append(Paragraph("7.9 Teacher-Student Consistency Learning", h2_style))
    story.append(Paragraph("The system maintains two parallel network instances: the Student network (updated via backpropagation) and the Teacher network (updated via Exponential Moving Average of Student weights). The Teacher generates pseudo-labels for unlabeled images, driving consistency learning.", body_style))

    story.append(Paragraph("7.10 Loss Functions (BCE + Soft Dice + Consistency MSE)", h2_style))
    story.append(Paragraph("On labeled patches, the supervised loss is a combination of Binary Cross-Entropy (BCE) and Soft Dice loss: <i>L<sub>sup</sub> = L<sub>BCE</sub> + L<sub>Dice</sub></i>. On unlabeled patches, an unsupervised Mean Squared Error (MSE) consistency loss is computed between Student predictions and Teacher pseudo-labels.", body_style))

    story.append(Paragraph("7.11 Teacher EMA Update & Buffer Remediation", h2_style))
    story.append(Paragraph("Teacher weights are updated with momentum &alpha; = 0.99: <i>&theta;<sub>T</sub> = &alpha; &theta;<sub>T</sub> + (1 - &alpha;) &theta;<sub>S</sub></i>. In our remediated implementation, Teacher BatchNorm buffers (running mean and variance) are also explicitly synchronized with Student buffers at each step, preventing activation divergence.", body_style))

    story.append(Paragraph("7.12 Numerical Instability Analysis (EXP-MLUA-002)", h2_style))
    story.append(Paragraph("During EXP-MLUA-002, training failed due to NaN values in Teacher GroupNorm layers. Forensic investigation revealed that because the Teacher ran in eval mode, its BatchNorm running buffers were never updated. When combined with EMA-updated weights, this caused extreme internal activation scaling (~10<sup>4</sup>), triggering overflow. Synchronizing buffers in EXP-MLUA-003 fully eliminated this failure.", body_style))

    story.append(Paragraph("7.13 Monte Carlo Uncertainty Estimation & Confidence Masking", h2_style))
    story.append(Paragraph("For each unlabeled patch, the Teacher performs T = 8 stochastic forward passes under dropout/perturbation. Predictive uncertainty is estimated as pixel-wise variance. A binary confidence mask <i>M(i, j)</i> filters out high-uncertainty pixels from the consistency loss, preventing the Student from learning noisy pseudo-labels.", body_style))

    story.append(Paragraph("7.14 Threshold Selection (Validation Sweep)", h2_style))
    story.append(Paragraph("A validation-only sensitivity sweep evaluated decision thresholds from &tau; = 0.05 to 0.95. Threshold <b>&tau; = 0.50</b> yielded the highest validation Dice (65.623%) with balanced Precision and Recall, and was retained as the frozen operating threshold.", body_style))

    story.append(Paragraph("7.15 Sliding-Window Panoramic Reconstruction", h2_style))
    story.append(Paragraph("During full-image inference, 21 overlapping 384&times;384 patch predictions are stitched back into the 768&times;1536 coordinate grid, with overlapping areas averaged via linear blending.", body_style))

    story.append(Paragraph("7.16 Final Output Representation", h2_style))
    story.append(Paragraph("The system outputs a continuous caries probability map, a thresholded binary mask (&tau; = 0.50), and bounding coordinates highlighting suspected carious regions for clinical review.", body_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 8: IMPLEMENTATION MODULES
    # ==========================================
    story.append(Paragraph("8. SYSTEM IMPLEMENTATION MODULES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    story.append(Paragraph("The project is structured into modular PyTorch components as detailed in Table 2.", body_style))

    t2_data = [
        [Paragraph("<b>Module Name</b>", table_header_style), Paragraph("<b>Implementation File</b>", table_header_style), Paragraph("<b>Core Functional Responsibility</b>", table_header_style)],
        [Paragraph("8.1 Dataset & Loader", table_cell_bold), Paragraph("src/mlua/data/dataset.py", table_cell_style), Paragraph("Loads DC1000 OPGs, applies normalization, extracts 384x384 patches.", table_cell_style)],
        [Paragraph("8.2 Patch Generator", table_cell_bold), Paragraph("src/mlua/data/patch_extractor.py", table_cell_style), Paragraph("Extracts 21 patches per 768x1536 image with stride S=192.", table_cell_style)],
        [Paragraph("8.3 ResNet-34 Encoder", table_cell_bold), Paragraph("src/mlua/models/encoder.py", table_cell_style), Paragraph("Extracts hierarchical visual features across stages C2-C5.", table_cell_style)],
        [Paragraph("8.4 FPN Decoder", table_cell_bold), Paragraph("src/mlua/models/fpn.py", table_cell_style), Paragraph("Fuses multi-scale features; hosts 4 auxiliary + 1 fused heads.", table_cell_style)],
        [Paragraph("8.5 Teacher-Student Engine", table_cell_bold), Paragraph("src/mlua/models/teacher_student.py", table_cell_style), Paragraph("Manages dual models, EMA weight updates, and buffer sync.", table_cell_style)],
        [Paragraph("8.6 Uncertainty Estimator", table_cell_bold), Paragraph("src/mlua/uncertainty/mc_sampler.py", table_cell_style), Paragraph("Performs T=8 Monte Carlo passes; generates confidence masks.", table_cell_style)],
        [Paragraph("8.7 Loss Manager", table_cell_bold), Paragraph("src/mlua/losses/combined_loss.py", table_cell_style), Paragraph("Computes BCE, Soft Dice, and masked consistency MSE losses.", table_cell_style)],
        [Paragraph("8.8 Training Trainer", table_cell_bold), Paragraph("src/mlua/trainers/trainer.py", table_cell_style), Paragraph("Orchestrates 60-epoch loop, AdamW optimizer, and Cosine scheduler.", table_cell_style)],
        [Paragraph("8.9 Inference & Stitching", table_cell_bold), Paragraph("src/mlua/inference/reconstruction.py", table_cell_style), Paragraph("Executes sliding-window inference and blends full OPG masks.", table_cell_style)],
        [Paragraph("8.10 Evaluation Metrics", table_cell_bold), Paragraph("src/mlua/metrics/evaluator.py", table_cell_style), Paragraph("Computes macro and micro Dice, IoU, Precision, Recall, Specificity.", table_cell_style)],
    ]
    t2 = Table(t2_data, colWidths=[120, 140, 220])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t2)
    story.append(Paragraph("<b>Table 2:</b> Summary of modular software implementation components in the codebase.", caption_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 9: EXPERIMENTAL PROGRESS & RESULTS
    # ==========================================
    story.append(Paragraph("9. EXPERIMENTAL PROGRESS AND RESULTS", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    story.append(Paragraph("9.1 Published MLUA Baseline Context", h2_style))
    story.append(Paragraph("The baseline paper (Wang et al., <i>Neurocomputing</i> 2023) established that multi-level uncertainty estimation can effectively guide semi-supervised segmentation on panoramic dental radiographs. We adopted this core concept while adapting the implementation to our standardized DC1000 dataset partition and modern PyTorch environment.", body_style))

    story.append(Paragraph("9.2 EXP-MLUA-001 (Baseline Run & Degradation)", h2_style))
    story.append(Paragraph("Our first experiment, <b>EXP-MLUA-001</b>, evaluated the initial baseline setup. While training completed, the model suffered from foreground suppression, predicting very few caries pixels due to extreme background dominance (validation Dice ~48%). This indicated the need for improved loss balancing and consistency scaling.", body_style))

    story.append(Paragraph("9.3 EXP-MLUA-002 (Numerical Instability & Collapse)", h2_style))
    story.append(Paragraph("In <b>EXP-MLUA-002</b>, we scaled the consistency weight and refined patch augmentations. However, training collapsed abruptly in early epochs with non-finite (NaN) loss values during the Teacher forward pass.", body_style))

    story.append(Paragraph("9.4 Forensic Diagnostic Analysis of EXP-MLUA-002", h2_style))
    story.append(Paragraph("A deep diagnostic audit traced the NaN failure to the Teacher network's GroupNorm layers. Because the Teacher was evaluated in eval mode, its BatchNorm running buffers were never updated during training. Meanwhile, its weights were continuously updated via EMA from the Student. This disparity between weights and uninitialized buffer statistics caused internal activations to explode to extreme values (~10<sup>4</sup>), overflowing numerical precision.", body_style))

    story.append(Paragraph("9.5 Remediation: Dual Parameter & Buffer EMA Synchronization", h2_style))
    story.append(Paragraph("To remediate the instability, we modified the EMA update routine to synchronize both model parameters and BatchNorm running statistics: <i>t_buf = &alpha; &middot; t_buf + (1 - &alpha;) &middot; s_buf</i>. Figure 3 illustrates the comparative stability between EXP-002 and EXP-003.", body_style))

    fig3_p = os.path.join("outputs", "progress_figures", "fig3_failure_remediation.png")
    if os.path.exists(fig3_p):
        story.append(Image(fig3_p, width=480, height=210))
        story.append(Paragraph("<b>Figure 3:</b> Comparison of internal model synchronization states: EXP-MLUA-002 (missing buffer sync causing collapse) vs. EXP-MLUA-003 (remediated dual sync ensuring 60-epoch stability).", caption_style))

    story.append(Paragraph("9.6 EXP-MLUA-003 (Corrected 60-Epoch Run)", h2_style))
    story.append(Paragraph("With buffer synchronization implemented, <b>EXP-MLUA-003</b> completed 60 epochs (7,920 global steps) without any numerical instability. Table 3 presents validation milestones across training.", body_style))

    t3_data = [
        [Paragraph("<b>Epoch (Step)</b>", table_header_style), Paragraph("<b>Val Loss</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>Precision (%)</b>", table_header_style), Paragraph("<b>Recall (%)</b>", table_header_style), Paragraph("<b>Observation / Note</b>", table_header_style)],
        [Paragraph("Epoch 01 (Step 132)", table_cell_bold), Paragraph("1.2415", table_cell_style), Paragraph("41.21%", table_cell_style), Paragraph("45.12%", table_cell_style), Paragraph("37.93%", table_cell_style), Paragraph("Initial baseline convergence", table_cell_style)],
        [Paragraph("Epoch 20 (Step 2640)", table_cell_bold), Paragraph("0.9124", table_cell_style), Paragraph("57.84%", table_cell_style), Paragraph("61.23%", table_cell_style), Paragraph("54.81%", table_cell_style), Paragraph("Steady progressive learning", table_cell_style)],
        [Paragraph("Epoch 40 (Step 5280)", table_cell_bold), Paragraph("0.8045", table_cell_style), Paragraph("63.12%", table_cell_style), Paragraph("66.45%", table_cell_style), Paragraph("60.10%", table_cell_style), Paragraph("Consistent boundary delineation", table_cell_style)],
        [Paragraph("Epoch 50 (Step 6600)", table_cell_bold), Paragraph("0.7782", table_cell_style), Paragraph("64.89%", table_cell_style), Paragraph("68.12%", table_cell_style), Paragraph("62.02%", table_cell_style), Paragraph("High precision stability", table_cell_style)],
        [Paragraph("<b>Epoch 56 (Step 7392)*</b>", table_cell_bold), Paragraph("<b>0.7639</b>", table_cell_bold), Paragraph("<b>65.62%</b>", table_cell_bold), Paragraph("<b>69.01%</b>", table_cell_bold), Paragraph("<b>63.65%</b>", table_cell_bold), Paragraph("<b>Optimal peak validation checkpoint</b>", table_cell_bold)],
        [Paragraph("Epoch 60 (Step 7920)", table_cell_bold), Paragraph("0.7712", table_cell_style), Paragraph("64.92%", table_cell_style), Paragraph("68.21%", table_cell_style), Paragraph("61.94%", table_cell_style), Paragraph("Slight post-peak plateau", table_cell_style)],
    ]
    t3 = Table(t3_data, colWidths=[95, 60, 65, 65, 65, 130])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('BACKGROUND', (0,5), (-1,5), c_alert_bg),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t3)
    story.append(Paragraph("<b>Table 3:</b> EXP-MLUA-003 validation milestone progression (*Epoch 56 selected as final model).", caption_style))

    story.append(Paragraph("9.7 Best Checkpoint Selection (Epoch 56)", h2_style))
    story.append(Paragraph("Among all evaluated epochs, <b>Epoch 56 (EXP-MLUA-003_E56_FINAL.pth)</b> achieved the lowest validation loss (0.76388) and highest validation Dice (65.623%), establishing it as our frozen primary model.", body_style))

    story.append(Paragraph("9.8 Validation Threshold Sensitivity Analysis", h2_style))
    story.append(Paragraph("Using checkpoint E56, a threshold sensitivity sweep across &tau; &isin; [0.05, 0.95] confirmed that <b>&tau; = 0.50</b> provides the optimal trade-off between Precision and Recall on validation data (Figure 4 & Table 4).", body_style))

    fig4_p = os.path.join("outputs", "progress_figures", "fig4_threshold_sensitivity.png")
    if os.path.exists(fig4_p):
        story.append(Image(fig4_p, width=440, height=210))
        story.append(Paragraph("<b>Figure 4:</b> Validation threshold sensitivity curve on checkpoint E56 across thresholds &tau; = 0.05 to 0.95.", caption_style))

    t4_data = [
        [Paragraph("<b>Threshold (&tau;)</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>Threshold (&tau;)</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style), Paragraph("<b>Threshold (&tau;)</b>", table_header_style), Paragraph("<b>Val Dice (%)</b>", table_header_style)],
        [Paragraph("&tau; = 0.10", table_cell_style), Paragraph("61.86%", table_cell_style), Paragraph("&tau; = 0.40", table_cell_style), Paragraph("65.42%", table_cell_style), Paragraph("&tau; = 0.70", table_cell_style), Paragraph("65.23%", table_cell_style)],
        [Paragraph("&tau; = 0.20", table_cell_style), Paragraph("64.02%", table_cell_style), Paragraph("<b>&tau; = 0.50*</b>", table_cell_bold), Paragraph("<b>65.62%</b>", table_cell_bold), Paragraph("&tau; = 0.80", table_cell_style), Paragraph("64.17%", table_cell_style)],
        [Paragraph("&tau; = 0.30", table_cell_style), Paragraph("64.93%", table_cell_style), Paragraph("&tau; = 0.60", table_cell_style), Paragraph("65.52%", table_cell_style), Paragraph("&tau; = 0.90", table_cell_style), Paragraph("61.40%", table_cell_style)],
    ]
    t4 = Table(t4_data, colWidths=[80, 80, 80, 80, 80, 80])
    t4.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t4)
    story.append(Paragraph("<b>Table 4:</b> Compact validation threshold sweep summary (*&tau; = 0.50 selected).", caption_style))

    story.append(Paragraph("9.9 Independent Sealed Test Evaluation (100 OPGs)", h2_style))
    story.append(Paragraph("The frozen E56 model was evaluated on an independent sealed test set of 100 full panoramic images (117,964,800 pixels) using sliding-window reconstruction at &tau; = 0.50. The model achieved <b>Macro Dice = 43.041%</b>, <b>Macro IoU = 29.057%</b>, <b>Macro Precision = 41.244%</b>, <b>Macro Recall = 52.896%</b>, and <b>Macro Specificity = 99.630%</b>, with <b>Micro Dice = 43.391%</b> (TP = 263,935, FP = 434,392, FN = 254,282, TN = 117,012,191).", body_style))

    story.append(Paragraph("9.10 Validation vs. Sealed Test Generalization Gap", h2_style))
    story.append(Paragraph("Table 5 and Figure 5 compare validation performance with sealed test results. The -22.58% drop in Dice is primarily driven by false-positive predictions on background anatomical artifacts (cervical burnout and restorations), highlighting the primary focus for our remaining work.", body_style))

    fig5_p = os.path.join("outputs", "progress_figures", "fig5_val_vs_test.png")
    if os.path.exists(fig5_p):
        story.append(Image(fig5_p, width=440, height=210))
        story.append(Paragraph("<b>Figure 5:</b> Performance comparison across Validation (patches) vs. Independent Sealed Test (full OPGs).", caption_style))

    t5_data = [
        [Paragraph("<b>Performance Metric</b>", table_header_style), Paragraph("<b>Validation (E56 Patches)</b>", table_header_style), Paragraph("<b>Sealed Test (Macro OPG)</b>", table_header_style), Paragraph("<b>Difference (&Delta;)</b>", table_header_style), Paragraph("<b>Primary Diagnostic Cause</b>", table_header_style)],
        [Paragraph("Dice Similarity", table_cell_bold), Paragraph("65.623%", table_cell_style), Paragraph("43.041%", table_cell_style), Paragraph("-22.582 pp", table_cell_style), Paragraph("Full-image background artifacts & stitching", table_cell_style)],
        [Paragraph("Intersection-over-Union", table_cell_bold), Paragraph("49.854%", table_cell_style), Paragraph("29.057%", table_cell_style), Paragraph("-20.797 pp", table_cell_style), Paragraph("Compounded boundary penalty on OPG canvas", table_cell_style)],
        [Paragraph("Precision", table_cell_bold), Paragraph("69.009%", table_cell_style), Paragraph("41.244%", table_cell_style), Paragraph("-27.765 pp", table_cell_style), Paragraph("Cervical burnout & restoration false positives", table_cell_style)],
        [Paragraph("Recall", table_cell_bold), Paragraph("63.649%", table_cell_style), Paragraph("52.896%", table_cell_style), Paragraph("-10.753 pp", table_cell_style), Paragraph("Incipient non-cavitated demineralization misses", table_cell_style)],
        [Paragraph("Specificity", table_cell_bold), Paragraph("99.753%", table_cell_style), Paragraph("99.630%", table_cell_style), Paragraph("-0.123 pp", table_cell_style), Paragraph("Consistently high background rejection", table_cell_style)],
    ]
    t5 = Table(t5_data, colWidths=[95, 90, 90, 75, 130])
    t5.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
    ]))
    story.append(t5)
    story.append(Paragraph("<b>Table 5:</b> Direct comparison between validation performance and independent sealed test evaluation.", caption_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 10: CURRENT STATUS & WORK REMAINING
    # ==========================================
    story.append(Paragraph("10. CURRENT STATUS AND WORK REMAINING", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    story.append(Paragraph("To provide the mentor with an accurate overview of current project progress and remaining deliverables, Table 6 outlines component statuses, while Table 7 specifies the remaining tasks.", body_style))

    t6_data = [
        [Paragraph("<b>System Component / Milestone</b>", table_header_style), Paragraph("<b>Current Implementation Status</b>", table_header_style), Paragraph("<b>Verification Artifact / Evidence</b>", table_header_style)],
        [Paragraph("DC1000 Dataset Preparation", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("2,389 training patches (530 L / 1,859 U)", table_cell_style)],
        [Paragraph("Baseline MLUA Implementation", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("src/mlua/ codebase functional", table_cell_style)],
        [Paragraph("EXP-MLUA-001 & 002 Trials", table_cell_bold), Paragraph("Completed / Historical", table_cell_style), Paragraph("Documented degradation and NaN crash", table_cell_style)],
        [Paragraph("Forensic Buffer Remediation", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("Dual weight + BN buffer EMA sync implemented", table_cell_style)],
        [Paragraph("EXP-MLUA-003 Training (60 Epochs)", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("outputs/experiments/EXP-MLUA-003_FINAL/", table_cell_style)],
        [Paragraph("Validation Checkpoint Selection", table_cell_bold), Paragraph("Completed (Epoch 56)", table_cell_style), Paragraph("Val Dice = 65.623% @ &tau; = 0.50", table_cell_style)],
        [Paragraph("Validation Threshold Sensitivity", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("19 thresholds evaluated (&tau; = 0.05 to 0.95)", table_cell_style)],
        [Paragraph("Sealed Test Evaluation (100 OPGs)", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("Macro Dice = 43.041%, Micro = 43.391%", table_cell_style)],
        [Paragraph("Full-Image Inference Pipeline", table_cell_bold), Paragraph("Completed", table_cell_style), Paragraph("Sliding-window reconstruction verified", table_cell_style)],
        [Paragraph("Application UI / Interface", table_cell_bold), Paragraph("In Progress", table_cell_style), Paragraph("Web UI demo active in repository", table_cell_style)],
        [Paragraph("False-Positive Reduction & Tuning", table_cell_bold), Paragraph("Pending / In Progress", table_cell_style), Paragraph("Target for next capstone sprint", table_cell_style)],
    ]
    t6 = Table(t6_data, colWidths=[140, 110, 230])
    t6.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_secondary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t6)
    story.append(Paragraph("<b>Table 6:</b> Current component implementation status across the capstone project.", caption_style))
    story.append(Spacer(1, 6))

    t7_data = [
        [Paragraph("<b>Remaining Task</b>", table_header_style), Paragraph("<b>Detailed Technical Scope</b>", table_header_style), Paragraph("<b>Target Objective / Metric</b>", table_header_style)],
        [Paragraph("1. False-Positive Filtering", table_cell_bold), Paragraph("Implement post-processing morphological filters to remove cervical burnout artifacts.", table_cell_style), Paragraph("Improve sealed test Precision from 41.2% to >50%.", table_cell_style)],
        [Paragraph("2. Test Error Categorization", table_cell_bold), Paragraph("Conduct systematic error auditing on the lowest 10% test cases to isolate failure modes.", table_cell_style), Paragraph("Identify dominant anatomical failure patterns.", table_cell_style)],
        [Paragraph("3. UI / Application Polish", table_cell_bold), Paragraph("Complete seamless integration between frozen E56 backend and interactive front-end.", table_cell_style), Paragraph("Enable real-time drag-and-drop OPG analysis.", table_cell_style)],
        [Paragraph("4. Final Documentation", table_cell_bold), Paragraph("Compile final capstone report, code documentation, and presentation slide deck.", table_cell_style), Paragraph("Final project defense readiness.", table_cell_style)],
    ]
    t7 = Table(t7_data, colWidths=[120, 210, 150])
    t7.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.white, c_light_bg]),
    ]))
    story.append(t7)
    story.append(Paragraph("<b>Table 7:</b> Remaining capstone project deliverables and target objectives.", caption_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 11: CONCLUSION
    # ==========================================
    story.append(Paragraph("11. CONCLUSION", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))
    
    conc_text = "This capstone project began with the study and baseline implementation of the published MLUA semi-supervised learning approach for dental panoramic caries segmentation. Initial experimental trials revealed significant challenges: EXP-MLUA-001 showed severe foreground degradation, while EXP-MLUA-002 encountered catastrophic numerical instability during Teacher inference. A thorough forensic investigation identified the absence of Teacher BatchNorm buffer synchronization during EMA updates as the core issue. Correcting this in <b>EXP-MLUA-003</b> enabled a stable 60-epoch training run. The resulting <b>Epoch 56 checkpoint</b> achieved <b>65.623% Dice</b> on validation data and <b>43.041% Macro Dice</b> (52.896% Recall, 99.630% Specificity) on an independent sealed test set of 100 full panoramic images. These results establish substantial progress while clearly delineating the remaining work required in false-positive reduction, error analysis, and application finalization."
    story.append(Paragraph(conc_text, body_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 12: FUTURE SCOPE
    # ==========================================
    story.append(Paragraph("12. FUTURE SCOPE", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))
    
    fs_items = [
        "1. <b>Anatomical False-Positive Filtering:</b> Integrate dedicated tooth-boundary segmentation priors to mask out non-dentition background areas.",
        "2. <b>Boundary-Aware Loss Regularization:</b> Explore boundary-weighted loss functions (e.g., Boundary IoU or Hausdorff loss) to sharpen diffuse demineralization contours.",
        "3. <b>Multi-Center Cross-Scanner Validation:</b> Evaluate model robustness across radiographs from different imaging hardware and clinical institutions.",
        "4. <b>Transformer-Based Encoders:</b> Investigate Swin Transformer backbones to capture global bilateral dental arch symmetry.",
        "5. <b>Interactive Clinical Workflow:</b> Develop an interactive review interface enabling dental practitioners to adjust sensitivity thresholds in real time."
    ]
    for fs in fs_items:
        story.append(Paragraph(fs, bullet_style))
    story.append(Spacer(1, 8))

    # ==========================================
    # SECTION 13: REFERENCES
    # ==========================================
    story.append(Paragraph("13. REFERENCES", h1_style))
    story.append(HRFlowable(width="100%", thickness=1, color=c_primary, spaceAfter=8))

    refs = [
        "[1] X. Wang, S. Gao, K. Jiang, H. Zhang, L. Wang, F. Chen, J. Yu, and F. Yang, \"Multi-level uncertainty aware learning for semi-supervised dental panoramic caries segmentation,\" <i>Neurocomputing</i>, vol. 540, p. 126208, 2023. DOI: 10.1016/j.neucom.2023.03.069.",
        "[2] X. Wang, S. Gao, et al., \"Official MLUA Research Codebase,\" GitHub Repository: https://github.com/Zzz512/MLUA, 2023.",
        "[3] K. He, X. Zhang, S. Ren, and J. Sun, \"Deep residual learning for image recognition,\" in <i>Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)</i>, 2016, pp. 770-778.",
        "[4] T.-Y. Lin, P. Doll&aacute;r, R. Girshick, K. He, B. Hariharan, and S. Belongie, \"Feature pyramid networks for object detection,\" in <i>Proc. IEEE Conf. Computer Vision and Pattern Recognition (CVPR)</i>, 2017, pp. 2117-2125.",
        "[5] A. Tarvainen and H. Valpola, \"Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results,\" in <i>Advances in Neural Information Processing Systems (NeurIPS)</i>, vol. 30, 2017.",
        "[6] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,\" in <i>International Conference on Machine Learning (ICML)</i>, 2016, pp. 1050-1059.",
        "[7] F. Milletari, N. Navab, and S.-A. Ahmadi, \"V-Net: Fully convolutional neural networks for volumetric medical image segmentation,\" in <i>Fourth International Conference on 3D Vision (3DV)</i>, 2016, pp. 565-571.",
        "[8] I. Loshchilov and F. Hutter, \"Decoupled weight decay regularization,\" in <i>International Conference on Learning Representations (ICLR)</i>, 2019."
    ]
    for r in refs:
        story.append(Paragraph(r, ParagraphStyle('RefP', parent=body_style, leftIndent=16, firstLineIndent=-12, spaceAfter=3)))

    doc.build(story, canvasmaker=ProgressNumberedCanvas)
    print(f"Successfully generated Capstone Progress Report: {output_filename}")

if __name__ == "__main__":
    create_progress_pdf()
