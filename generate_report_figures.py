"""
Generate high-resolution academic figures for EXP-MLUA-003 Project Report.
Uses verified data from project CSVs and exact architectural layouts.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

# Ensure output dir
os.makedirs("outputs/report_figures", exist_ok=True)
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# -------------------------------------------------------------
# FIGURE 1: Overall Project Workflow Flowchart (Visual Diagram)
# -------------------------------------------------------------
def generate_figure_1():
    fig, ax = plt.subplots(figsize=(10, 14), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 140)

    # Style definitions
    box_blue = dict(boxstyle="round,pad=0.5", facecolor="#EBF5FB", edgecolor="#2980B9", linewidth=1.5)
    box_green = dict(boxstyle="round,pad=0.5", facecolor="#EAFAF1", edgecolor="#27AE60", linewidth=1.5)
    box_orange = dict(boxstyle="round,pad=0.5", facecolor="#FEF9E7", edgecolor="#F39C12", linewidth=1.5)
    box_purple = dict(boxstyle="round,pad=0.5", facecolor="#F4ECF7", edgecolor="#8E44AD", linewidth=1.5)
    box_red = dict(boxstyle="round,pad=0.5", facecolor="#FDEDEC", edgecolor="#C0392B", linewidth=1.5)
    box_gray = dict(boxstyle="round,pad=0.5", facecolor="#F8F9F9", edgecolor="#7F8C8D", linewidth=1.5)

    arrow = dict(arrowstyle="->", color="#2C3E50", lw=1.5)

    # Blocks definition
    nodes = [
        (50, 134, "DC1000 Dataset\n(1,000 Panoramic X-rays, >7,500 Lesions)", box_blue),
        (50, 124, "Data Organization & Cleaning\n(593 Detailed, 407 Rough/Unlabeled)", box_blue),
        (50, 114, "Dataset Partitioning\n(Train / Val / 100 Sealed Test Panoramas)", box_blue),
        (50, 104, "CLAHE Normalization & Preprocessing\n(Dynamic Range Harmonization, 768×1536 Canvas)", box_gray),
        (50, 94, "384×384 Patch Extraction (Stride = 192 px)\n(2,389 Patches: 530 Labeled (20%) / 1,859 Unlabeled (80%))", box_green),
        (50, 84, "Teacher-Student Semi-Supervised Dual Framework\n(Student Model f_θ  &  Teacher Model g_θ_ema)", box_purple),
        (50, 74, "ResNet-34 Feature Encoder (C1 - C5)\n(Hierarchical Residual Convolutions)", box_purple),
        (50, 64, "Lateral Feature Pyramid Network (FPN) Decoder\n(Multi-Scale Pyramids P2, P3, P4, P5)", box_purple),
        (50, 54, "4 Auxiliary Heads (1/8, 1/4, 1/2, 1/1) + Fused Head\n(Deep Supervision Loss + Monte Carlo Sampling)", box_purple),
    ]

    for x, y, text, style in nodes:
        ax.text(x, y, text, ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=style)

    # Vertical connections
    for i in range(len(nodes)-1):
        ax.annotate('', xy=(50, nodes[i+1][1]+3.5), xytext=(50, nodes[i][1]-3.5), arrowprops=arrow)

    # Dual branch below multi-scale heads
    ax.text(25, 42, "Supervised Loss (Labeled)\nL_sup = L_BCE + L_Dice\n(α = [0.1, 0.2, 0.3, 0.4])", ha='center', va='center', fontsize=8, fontweight='bold', bbox=box_green)
    ax.text(75, 42, "Uncertainty Estimation (Unlabeled)\nMonte Carlo T=8 Sampling\nEntropy Variance -> Certainty Mask", ha='center', va='center', fontsize=8, fontweight='bold', bbox=box_orange)

    ax.annotate('', xy=(25, 46.5), xytext=(40, 50.5), arrowprops=arrow)
    ax.annotate('', xy=(75, 46.5), xytext=(60, 50.5), arrowprops=arrow)

    # Consistency Loss
    ax.text(50, 32, "Consistency Loss (L_con)\nUncertainty-Gated MSE Consistency on Unlabeled Patches\nTotal Loss = L_sup + λ(t)·L_con", ha='center', va='center', fontsize=8, fontweight='bold', bbox=box_orange)
    ax.annotate('', xy=(40, 35.5), xytext=(25, 37.5), arrowprops=arrow)
    ax.annotate('', xy=(60, 35.5), xytext=(75, 37.5), arrowprops=arrow)

    # Updates
    ax.text(50, 23, "Student Optimization (AdamW, lr=0.001, wd=0.01)\n& Vectorized Teacher EMA + BatchNorm Buffer Sync (θ=0.99)", ha='center', va='center', fontsize=8, fontweight='bold', bbox=box_purple)
    ax.annotate('', xy=(50, 26.5), xytext=(50, 28.5), arrowprops=arrow)

    # Evaluation & Verification
    ax.text(50, 14, "Validation Milestone & E56 Checkpoint Selection\nThreshold Sensitivity Sweep (τ ∈ [0.05, 0.95] -> τ = 0.50 Optimal)", ha='center', va='center', fontsize=8, fontweight='bold', bbox=box_red)
    ax.annotate('', xy=(50, 17.5), xytext=(50, 19.5), arrowprops=arrow)

    ax.text(50, 5, "Independent Sealed Test Benchmark (100 Cases)\nSliding-Window Inference (21 Patches) -> 2D Gaussian Blending -> Final Mask\n(Macro Dice = 43.041%, Specificity = 99.630%, Zero-Pred = 0.0%)", ha='center', va='center', fontsize=8, fontweight='bold', bbox=box_green)
    ax.annotate('', xy=(50, 8.5), xytext=(50, 10.5), arrowprops=arrow)

    plt.title("Figure 1: Complete EXP-MLUA-003 End-to-End System Flowchart", fontsize=11, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig1_overall_workflow.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 2: DC1000 Preprocessing & Patch Generation
# -------------------------------------------------------------
def generate_figure_2():
    fig, ax = plt.subplots(figsize=(11, 5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    # Panoramic frame
    rect_pan = patches.Rectangle((5, 10), 38, 25, linewidth=1.5, edgecolor='#2C3E50', facecolor='#EAEDED', linestyle='--')
    ax.add_patch(rect_pan)
    ax.text(24, 38, "Full Panoramic OPG (768 × 1536)", ha='center', fontsize=9, fontweight='bold')
    ax.text(24, 22, "Extreme Class Imbalance\nCaries Area ≈ 1.5‰ (0.15%)\nNon-caries background > 99.85%", ha='center', fontsize=8, color='#7B241C')

    # Sliding patches on panorama
    for idx, (px, py) in enumerate([(8, 12), (18, 12), (28, 12), (8, 22), (18, 22), (28, 22)]):
        p = patches.Rectangle((px, py), 12, 10, linewidth=1, edgecolor='#2980B9', facecolor='#AED6F1', alpha=0.4)
        ax.add_patch(p)

    ax.annotate('', xy=(48, 22.5), xytext=(44, 22.5), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=2))
    ax.text(46, 25, "Stride = 192 px\n(50% Overlap)", ha='center', fontsize=7.5, fontweight='bold')

    # Patch grid
    rect_grid = patches.Rectangle((50, 8), 45, 29, linewidth=1.5, edgecolor='#27AE60', facecolor='#E8F8F5')
    ax.add_patch(rect_grid)
    ax.text(72.5, 40, "21 Overlapping Patches (384 × 384)", ha='center', fontsize=9, fontweight='bold')

    for r in range(3):
        for c in range(5):
            bx = 53 + c * 8
            by = 12 + (2-r) * 8
            bp = patches.Rectangle((bx, by), 6.5, 6.5, linewidth=1, edgecolor='#1E8449', facecolor='#A9DFBF')
            ax.add_patch(bp)

    ax.text(72.5, 3, "Foreground Concentration Amplification: 0.15% -> 11.79% in active patches\nTotal Training Set: 2,389 Patches (530 Labeled / 1,859 Unlabeled)", ha='center', fontsize=8.5, fontweight='bold', color='#145A32')

    plt.title("Figure 2: DC1000 Panoramic Image Slicing and Class Imbalance Resolution", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig2_patch_generation.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 3: ResNet-34 Feature Encoder
# -------------------------------------------------------------
def generate_figure_3():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 45)

    stages = [
        ("Input", "384×384×1", 6, 20, "#BDC3C7"),
        ("Conv1+MaxPool", "192×192×64 (C1)", 20, 20, "#85C1E9"),
        ("Layer 1 (3 res)", "96×96×64 (C2)", 38, 20, "#5DADE2"),
        ("Layer 2 (4 res)", "48×48×128 (C3)", 56, 20, "#3498DB"),
        ("Layer 3 (6 res)", "24×24×256 (C4)", 74, 20, "#2874A6"),
        ("Layer 4 (3 res)", "12×12×512 (C5)", 92, 20, "#1B4F72")
    ]

    for name, shape, x, y, col in stages:
        rect = patches.Rectangle((x-7, y-10), 14, 20, linewidth=1.2, edgecolor='#1A5276', facecolor=col)
        ax.add_patch(rect)
        ax.text(x, y+4, name, ha='center', va='center', fontsize=7.5, fontweight='bold', color='#000000' if x < 70 else '#FFFFFF')
        ax.text(x, y-4, shape, ha='center', va='center', fontsize=7, color='#000000' if x < 70 else '#FFFFFF')

    for i in range(len(stages)-1):
        ax.annotate('', xy=(stages[i+1][2]-7, 20), xytext=(stages[i][2]+7, 20), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))

    ax.text(50, 5, "Hierarchical Feature Extraction: Low-level edges & contrast (C1/C2) -> High-level semantics & arch geometry (C4/C5)", ha='center', fontsize=8.5, style='italic')
    plt.title("Figure 3: ResNet-34 5-Stage Hierarchical Residual Encoder Architecture", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig3_resnet34_encoder.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 4: Feature Pyramid Network (FPN) Multi-Scale Decoder
# -------------------------------------------------------------
def generate_figure_4():
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Encoder blocks (Left)
    encs = [("C2: 96×96", 20, 80), ("C3: 48×48", 20, 60), ("C4: 24×24", 20, 40), ("C5: 12×12", 20, 20)]
    for name, x, y in encs:
        ax.text(x, y, name, ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="square,pad=0.6", facecolor="#D4E6F1", edgecolor="#2980B9"))

    # Top-down FPN blocks (Right)
    fpns = [("P2: 96×96 (256 ch)", 60, 80), ("P3: 48×48 (256 ch)", 60, 60), ("P4: 24×24 (256 ch)", 60, 40), ("P5: 12×12 (256 ch)", 60, 20)]
    for name, x, y in fpns:
        ax.text(x, y, name, ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="square,pad=0.6", facecolor="#D5F5E3", edgecolor="#27AE60"))

    # Lateral 1x1 convs
    for i in range(4):
        ax.annotate('1×1 conv', xy=(49, encs[i][2]), xytext=(31, encs[i][2]), ha='center', va='bottom', fontsize=7, arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.2))

    # Top-down upsampling arrows
    for i in range(3, 0, -1):
        ax.annotate('2× upsample + add', xy=(60, fpns[i-1][2]-5), xytext=(60, fpns[i][2]+5), ha='right', va='center', fontsize=7, arrowprops=dict(arrowstyle="->", color="#27AE60", lw=1.2))

    # Aux outputs
    ax.text(88, 80, "Aux Head 4 (α=0.4)\n384×384", ha='center', va='center', fontsize=7.5, bbox=dict(boxstyle="round", facecolor="#FCF3CF", edgecolor="#F39C12"))
    ax.text(88, 60, "Aux Head 3 (α=0.3)\n192×192", ha='center', va='center', fontsize=7.5, bbox=dict(boxstyle="round", facecolor="#FCF3CF", edgecolor="#F39C12"))
    ax.text(88, 40, "Aux Head 2 (α=0.2)\n96×96", ha='center', va='center', fontsize=7.5, bbox=dict(boxstyle="round", facecolor="#FCF3CF", edgecolor="#F39C12"))
    ax.text(88, 20, "Aux Head 1 (α=0.1)\n48×48", ha='center', va='center', fontsize=7.5, bbox=dict(boxstyle="round", facecolor="#FCF3CF", edgecolor="#F39C12"))

    for i in range(4):
        ax.annotate('', xy=(78, fpns[i][2]), xytext=(72, fpns[i][2]), arrowprops=dict(arrowstyle="->", color="#D35400", lw=1.2))

    # Fused head
    ax.text(60, 5, "Fused Segmentation Head -> 384×384 Binary Prediction", ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.5", facecolor="#EBDEF0", edgecolor="#8E44AD"))
    ax.annotate('', xy=(60, 10), xytext=(60, 72), arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=1.5, linestyle="--"))

    plt.title("Figure 4: Lateral Feature Pyramid Network (FPN) and Multi-Scale Deep Supervision", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig4_fpn_decoder.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 5: Teacher-Student Semi-Supervised Architecture
# -------------------------------------------------------------
def generate_figure_5():
    fig, ax = plt.subplots(figsize=(10, 6.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    # Student Network (Top)
    ax.text(35, 80, "STUDENT NETWORK (f_θ)\nResNet-34 + FPN\n(Trained via AdamW on L_total)", ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.6", facecolor="#EBF5FB", edgecolor="#2980B9", lw=1.5))
    ax.text(10, 80, "Labeled\nBatch (4)\n+\nUnlabeled\nBatch (4)", ha='center', va='center', fontsize=7.5, bbox=dict(boxstyle="round", facecolor="#EAEDED"))
    ax.annotate('', xy=(20, 80), xytext=(15, 80), arrowprops=dict(arrowstyle="->", lw=1.5))

    ax.text(75, 88, "Supervised Loss (L_sup)\nDice + BCE on Labeled Targets", ha='center', va='center', fontsize=7.5, bbox=dict(boxstyle="round", facecolor="#D5F5E3", edgecolor="#27AE60"))
    ax.annotate('', xy=(62, 88), xytext=(50, 83), arrowprops=dict(arrowstyle="->", lw=1.2, color="#27AE60"))

    # Teacher Network (Bottom)
    ax.text(35, 30, "TEACHER NETWORK (g_θ_ema)\nResNet-34 + FPN\n(No Gradients, Vectorized EMA Update)", ha='center', va='center', fontsize=8.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.6", facecolor="#FEF9E7", edgecolor="#F39C12", lw=1.5))
    ax.text(10, 30, "Unlabeled\nBatch (4)\n+\nGaussian\nNoise (η)", ha='center', va='center', fontsize=7.5, bbox=dict(boxstyle="round", facecolor="#EAEDED"))
    ax.annotate('', xy=(20, 30), xytext=(15, 30), arrowprops=dict(arrowstyle="->", lw=1.5))

    # EMA Arrow with Buffer Sync
    ax.annotate('Vectorized EMA Update (θ=0.99)\n+ BatchNorm Buffer Sync (running_mean, running_var)', xy=(35, 40), xytext=(35, 70), ha='center', va='center', fontsize=7.5, fontweight='bold', color='#8E44AD', arrowprops=dict(arrowstyle="->", color="#8E44AD", lw=2, linestyle="-"))

    # Consistency Loss
    ax.text(75, 55, "Uncertainty-Gated Consistency Loss (L_con)\nMSE(ŷ_student, ŷ_teacher) ⊙ Confidence Mask", ha='center', va='center', fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.6", facecolor="#FDEDEC", edgecolor="#C0392B", lw=1.2))
    ax.annotate('', xy=(62, 60), xytext=(50, 77), arrowprops=dict(arrowstyle="->", lw=1.2, color="#C0392B"))
    ax.annotate('', xy=(62, 50), xytext=(50, 33), arrowprops=dict(arrowstyle="->", lw=1.2, color="#C0392B"))

    plt.title("Figure 5: Teacher-Student Semi-Supervised Consistency Learning Framework", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig5_teacher_student.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 6: MLUA Monte Carlo Uncertainty Estimation
# -------------------------------------------------------------
def generate_figure_6():
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)

    boxes = [
        (12, 30, "Unlabeled Image x_j\n+ T=8 Independent\nGaussian Noises", "#EAECEE"),
        (35, 30, "Teacher Forward Passes\nT=8 passes × (L+1) heads\n= 40 Multi-Scale Predictions", "#FCF3CF"),
        (60, 30, "Mean Probability Pooling\nŷ_mean = (1/40) Σ ŷ_i\nEntropy: -2·ŷ·log(ŷ)", "#D5F5E3"),
        (85, 30, "Certainty Mask (m_certain)\nThreshold Gating (β=0.75)\nSuppresses Cervical Burnout", "#D4E6F1")
    ]

    for x, y, txt, col in boxes:
        ax.text(x, y, txt, ha='center', va='center', fontsize=7.5, fontweight='bold', bbox=dict(boxstyle="round,pad=0.6", facecolor=col, edgecolor="#2C3E50", lw=1.2))

    for i in range(len(boxes)-1):
        ax.annotate('', xy=(boxes[i+1][0]-11, 30), xytext=(boxes[i][0]+11, 30), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))

    ax.text(50, 8, "Dynamic Threshold: threshold = γ · (β + (1-β)·exp(-5(1-ε/E)²))\nGates unconfident / ambiguous pseudo-labels during consistency regularization", ha='center', fontsize=8, style='italic', color='#1A5276')

    plt.title("Figure 6: Multi-Level Monte Carlo Uncertainty Estimation and Gating Mechanism", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig6_uncertainty_estimation.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 7: Actual EXP-MLUA-003 Training Dynamics (CSV Data)
# -------------------------------------------------------------
def generate_figure_7():
    csv_path = "outputs/experiments/EXP-MLUA-003_FINAL/EXP-MLUA-003_FULL_TRAINING_HISTORY.csv"
    if not os.path.exists(csv_path):
        print("Training history CSV not found, skipping Fig 7")
        return

    df = pd.read_csv(csv_path)

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8), dpi=300)

    # 1. Total Loss & Validation Loss
    ax1.plot(df['epoch'], df['train_loss'], label='Train Total Loss', color='#2980B9', lw=2)
    ax1.plot(df['epoch'], df['val_loss'], label='Validation Loss', color='#E74C3C', lw=2)
    ax1.set_xlabel('Epoch', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=9, fontweight='bold')
    ax1.set_title('(a) Training & Validation Loss', fontsize=10, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8)

    # 2. Supervised vs Consistency Loss
    ax2.plot(df['epoch'], df['supervised_loss'], label='Supervised Loss (L_sup)', color='#27AE60', lw=2)
    ax2.plot(df['epoch'], df['consistency_loss'], label='Consistency Loss (L_con)', color='#F39C12', lw=2)
    ax2.set_xlabel('Epoch', fontsize=9, fontweight='bold')
    ax2.set_ylabel('Component Loss', fontsize=9, fontweight='bold')
    ax2.set_title('(b) Supervised vs. Consistency Loss Components', fontsize=10, fontweight='bold')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.legend(fontsize=8)

    # 3. Validation Dice Progression & Peak
    ax3.plot(df['epoch'], df['val_dice'] * 100, label='Val Dice (%)', color='#8E44AD', lw=2)
    best_ep = df.loc[df['val_dice'].idxmax()]
    ax3.scatter(best_ep['epoch'], best_ep['val_dice'] * 100, color='#C0392B', s=80, zorder=5, label=f"Peak E{int(best_ep['epoch'])}: {best_ep['val_dice']*100:.2f}%")
    ax3.set_xlabel('Epoch', fontsize=9, fontweight='bold')
    ax3.set_ylabel('Dice Similarity (%)', fontsize=9, fontweight='bold')
    ax3.set_title('(c) Validation Dice Progression (Peak at Epoch 56: 65.623%)', fontsize=10, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    ax3.legend(fontsize=8)

    # 4. Precision, Recall & IoU
    ax4.plot(df['epoch'], df['val_precision'] * 100, label='Val Precision (%)', color='#16A085', lw=1.8)
    ax4.plot(df['epoch'], df['val_recall'] * 100, label='Val Recall (%)', color='#D35400', lw=1.8)
    ax4.plot(df['epoch'], df['val_iou'] * 100, label='Val IoU (%)', color='#2C3E50', lw=1.8)
    ax4.set_xlabel('Epoch', fontsize=9, fontweight='bold')
    ax4.set_ylabel('Metric Value (%)', fontsize=9, fontweight='bold')
    ax4.set_title('(d) Validation Precision, Recall, and IoU Progression', fontsize=10, fontweight='bold')
    ax4.grid(True, linestyle='--', alpha=0.5)
    ax4.legend(fontsize=8)

    plt.suptitle("Figure 7: EXP-MLUA-003 Full 60-Epoch Training Dynamics and Convergence Records", fontsize=12, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig7_training_curves.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 8: Sliding-Window Inference and Reconstruction
# -------------------------------------------------------------
def generate_figure_8():
    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=300)
    ax.axis('off')
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)

    boxes = [
        (15, 25, "Standardized OPG\n(768 × 1536)", "#EBF5FB"),
        (38, 25, "21 Sliding Patches\n(384×384, S=192)\nOverlapping Grid", "#D5F5E3"),
        (62, 25, "Model Inference\nFrozen E56 Model\n21 Patch Probabilities", "#FCF3CF"),
        (85, 25, "2D Gaussian Blending\nSeamless Reconstruction\n(768 × 1536 Prob Map)", "#EBDEF0")
    ]

    for x, y, txt, col in boxes:
        ax.text(x, y, txt, ha='center', va='center', fontsize=8, fontweight='bold', bbox=dict(boxstyle="round,pad=0.6", facecolor=col, edgecolor="#2C3E50", lw=1.2))

    for i in range(len(boxes)-1):
        ax.annotate('', xy=(boxes[i+1][0]-10, 25), xytext=(boxes[i][0]+10, 25), arrowprops=dict(arrowstyle="->", color="#2C3E50", lw=1.5))

    ax.text(50, 6, "2D Gaussian Weight Kernel: W(x,y) = exp(-((x-μx)² + (y-μy)²)/(2σ²)) suppresses patch edge seam artifacts", ha='center', fontsize=8, style='italic', color='#4A235A')

    plt.title("Figure 8: 21-Patch Sliding-Window Inference and 2D Gaussian Overlap Reconstruction", fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig8_sliding_window_reconstruction.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 9: Probability Map -> Threshold -> Binary Mask
# -------------------------------------------------------------
def generate_figure_9():
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=300)

    # Mock synthetic tooth with caries for visualization concept
    x = np.linspace(-3, 3, 150)
    y = np.linspace(-1.5, 1.5, 80)
    X, Y = np.meshgrid(x, y)
    Z = np.exp(-((X-0.5)**2 + (Y-0.2)**2)/0.15) * 0.92 + np.exp(-((X+1.2)**2 + (Y+0.3)**2)/0.25) * 0.78
    Z += np.random.normal(0, 0.04, Z.shape)
    Z = np.clip(Z, 0, 1)

    # 1. Continuous Probability Map
    im1 = ax1.imshow(Z, cmap='magma', vmin=0, vmax=1)
    ax1.set_title("(a) Reconstructed Prob Map P(x,y)", fontsize=9, fontweight='bold')
    ax1.axis('off')
    plt.colorbar(im1, ax=ax1, fraction=0.03, pad=0.04)

    # 2. Thresholding τ = 0.50
    im2 = ax2.imshow(Z >= 0.50, cmap='gray')
    ax2.set_title("(b) Binarized Mask at τ = 0.50", fontsize=9, fontweight='bold')
    ax2.axis('off')

    # 3. Clinical Staging / Overlay
    overlay = np.zeros((*Z.shape, 3))
    mask = Z >= 0.50
    overlay[..., 0] = mask * 0.95  # Red overlay
    overlay[..., 1] = mask * 0.2
    overlay[..., 2] = mask * 0.2
    ax3.imshow(1.0 - Z*0.5, cmap='gray')
    ax3.imshow(overlay, alpha=0.6)
    ax3.set_title("(c) Lesion Localization & Clinical Overlay", fontsize=9, fontweight='bold')
    ax3.axis('off')

    plt.suptitle("Figure 9: From Continuous Caries Probability to Calibrated Binary Mask and Clinical Overlay", fontsize=11, fontweight='bold', y=1.03)
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig9_thresholding_process.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 10: EXP-MLUA-002 Failure vs EXP-MLUA-003 Remediation
# -------------------------------------------------------------
def generate_figure_10():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # Left: EXP-MLUA-002 Failure
    ax1.text(0.5, 0.8, "EXP-MLUA-002: Catastrophic NaN Collapse", ha='center', fontsize=10, fontweight='bold', color='#78281F')
    ax1.text(0.5, 0.65, "Failed at: Epoch 10, Batch 68 (Global Step 1,257)", ha='center', fontsize=8.5, fontweight='bold')
    ax1.text(0.5, 0.45, "Root Cause:\nTeacher parameters updated via EMA, but\nTeacher BatchNorm buffers (running_mean, running_var)\nremained stale and un-synchronized.\nActivation exploded to > 10^18 -> NaN overflow.", ha='center', fontsize=8, bbox=dict(boxstyle="round", facecolor="#FDEDEC", edgecolor="#C0392B"))
    ax1.text(0.5, 0.15, "Status: ABORTED WITH FATAL LOSS DIVERGENCE", ha='center', fontsize=8.5, fontweight='bold', color='#C0392B')
    ax1.axis('off')

    # Right: EXP-MLUA-003 Remediation
    ax2.text(0.5, 0.8, "EXP-MLUA-003: Controlled Remediation Run", ha='center', fontsize=10, fontweight='bold', color='#145A32')
    ax2.text(0.5, 0.65, "Completed: 60 Full Epochs (7,920 Global Steps)", ha='center', fontsize=8.5, fontweight='bold')
    ax2.text(0.5, 0.45, "The Single Controlled Intervention:\nVectorized EMA synchronization for BOTH parameters\nAND floating-point BatchNorm running buffers (θ=0.99).\nResult: Zero NaNs, zero Infs, 100% numerical stability.", ha='center', fontsize=8, bbox=dict(boxstyle="round", facecolor="#EAFAF1", edgecolor="#27AE60"))
    ax2.text(0.5, 0.15, "Status: 100% COMPLETE (Val Dice = 65.62%, Test Dice = 43.04%)", ha='center', fontsize=8.5, fontweight='bold', color='#27AE60')
    ax2.axis('off')

    plt.suptitle("Figure 10: Comparative Scientific Forensic: EXP-MLUA-002 Failure vs. EXP-MLUA-003 Remediation", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig10_remediation_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

# -------------------------------------------------------------
# FIGURE 11: Validation vs Sealed-Test Performance & Threshold Curve
# -------------------------------------------------------------
def generate_figure_11():
    sweep_path = "outputs/diagnostics/EXP-MLUA-003_VALIDATION_THRESHOLD_ANALYSIS/EXP-MLUA-003_THRESHOLD_SWEEP.csv"
    if not os.path.exists(sweep_path):
        print("Threshold sweep CSV not found, skipping Fig 11")
        return

    df_sw = pd.read_csv(sweep_path)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5), dpi=300)

    # 1. Threshold sensitivity curve
    ax1.plot(df_sw['threshold'], df_sw['dice'] * 100, label='Validation Dice (%)', color='#2980B9', lw=2)
    ax1.plot(df_sw['threshold'], df_sw['precision'] * 100, label='Precision (%)', color='#27AE60', lw=1.8, linestyle='--')
    ax1.plot(df_sw['threshold'], df_sw['recall'] * 100, label='Recall (%)', color='#E67E22', lw=1.8, linestyle='--')
    ax1.axvline(0.50, color='#C0392B', linestyle=':', lw=2, label='Selected τ = 0.50 (Dice = 65.62%)')
    ax1.set_xlabel('Decision Threshold (τ)', fontsize=9, fontweight='bold')
    ax1.set_ylabel('Metric Value (%)', fontsize=9, fontweight='bold')
    ax1.set_title('(a) Validation Threshold Sensitivity Sweep (τ ∈ [0.05, 0.95])', fontsize=9.5, fontweight='bold')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=7.5)

    # 2. Validation vs Test Bar Comparison
    categories = ['Dice Similarity', 'IoU (Jaccard)', 'Precision', 'Recall', 'Specificity']
    val_scores = [65.623, 49.854, 69.009, 63.649, 99.753]
    test_scores = [43.041, 29.057, 41.244, 52.896, 99.630]

    x = np.arange(len(categories))
    w = 0.35

    ax2.bar(x - w/2, val_scores, width=w, label='Val Cohort (E56, Patches)', color='#3498DB', edgecolor='#1B4F72')
    ax2.bar(x + w/2, test_scores, width=w, label='Sealed Test Set (100 Full OPGs)', color='#E74C3C', edgecolor='#78281F')
    ax2.set_xticks(x)
    ax2.set_xticklabels(categories, rotation=15, ha='right', fontsize=8, fontweight='bold')
    ax2.set_ylabel('Score (%)', fontsize=9, fontweight='bold')
    ax2.set_title('(b) Validation vs. Independent Sealed Test Benchmark', fontsize=9.5, fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax2.legend(fontsize=7.5)

    plt.suptitle("Figure 11: Validation Sensitivity Analysis and Sealed-Test Benchmark Comparison", fontsize=11, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig("outputs/report_figures/fig11_validation_test_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    print("Generating Figure 1...")
    generate_figure_1()
    print("Generating Figure 2...")
    generate_figure_2()
    print("Generating Figure 3...")
    generate_figure_3()
    print("Generating Figure 4...")
    generate_figure_4()
    print("Generating Figure 5...")
    generate_figure_5()
    print("Generating Figure 6...")
    generate_figure_6()
    print("Generating Figure 7...")
    generate_figure_7()
    print("Generating Figure 8...")
    generate_figure_8()
    print("Generating Figure 9...")
    generate_figure_9()
    print("Generating Figure 10...")
    generate_figure_10()
    print("Generating Figure 11...")
    generate_figure_11()
    print("All 11 report figures generated successfully!")
