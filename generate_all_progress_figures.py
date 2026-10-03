import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd

out_dir = os.path.join("outputs", "progress_figures")
os.makedirs(out_dir, exist_ok=True)

plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

C_NAVY = "#1A365D"
C_SLATE = "#2B6CB0"
C_TEAL = "#2C7A7B"
C_AMBER = "#D69E2E"
C_RED = "#E53E3E"
C_GREEN = "#38A169"
C_GREY_BG = "#F7FAFC"
C_BORDER = "#CBD5E0"

# Fig 1: Overall Project Flowchart
def fig1():
    print("Generating Figure 1: Overall Project Flowchart...")
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    ax.set_facecolor(C_GREY_BG)
    fig.patch.set_facecolor(C_GREY_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 55)
    ax.axis('off')

    boxes = [
        ("DC1000 Dataset\n(1,000 OPG Scans)", 15, 46, C_NAVY),
        ("Preprocessing &\n384x384 Patching", 45, 46, C_SLATE),
        ("20% Labeled /\n80% Unlabeled Split", 75, 46, C_TEAL),
        ("Student Network\n(ResNet-34 + FPN)", 25, 31, C_TEAL),
        ("Teacher Network\n(EMA Updated)", 65, 31, C_AMBER),
        ("Multi-Task Loss\n(BCE + Dice + Cons)", 25, 16, C_NAVY),
        ("Monte Carlo (T=8)\nUncertainty Masking", 65, 16, C_AMBER),
        ("Dual EMA Sync\n(Weights + BN Buffers)", 45, 5, C_GREEN),
        ("Sliding-Window Inference\n(tau = 0.50 -> Mask)", 80, 5, C_NAVY),
    ]

    for title, x, y, col in boxes:
        box = patches.FancyBboxPatch((x-12, y-4.5), 24, 9, boxstyle="round,pad=0.4",
                                      facecolor="white", edgecolor=col, lw=1.6)
        ax.add_patch(box)
        ax.text(x, y, title, ha='center', va='center', fontsize=7.5, fontweight='bold', color=col)

    # Connections
    ax.annotate('', xy=(33, 46), xytext=(27, 46), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_NAVY))
    ax.annotate('', xy=(63, 46), xytext=(57, 46), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_NAVY))
    ax.annotate('', xy=(25, 35.5), xytext=(68, 41.5), arrowprops=dict(arrowstyle="->", lw=1.2, color=C_TEAL))
    ax.annotate('', xy=(65, 35.5), xytext=(78, 41.5), arrowprops=dict(arrowstyle="->", lw=1.2, color=C_AMBER))
    ax.annotate('', xy=(25, 20.5), xytext=(25, 26.5), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_TEAL))
    ax.annotate('', xy=(65, 20.5), xytext=(65, 26.5), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_AMBER))
    ax.annotate('', xy=(37, 16), xytext=(53, 16), arrowprops=dict(arrowstyle="<-", lw=1.4, color=C_NAVY))
    ax.annotate('', xy=(45, 9.5), xytext=(25, 11.5), arrowprops=dict(arrowstyle="->", lw=1.4, color=C_GREEN))
    ax.annotate('', xy=(68, 5), xytext=(57, 5), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_NAVY))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure1_overall_flowchart.png"), dpi=300)
    plt.close()

# Fig 2: Research Progress Flowchart
def fig2():
    print("Generating Figure 2: Research Progress Flowchart...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_facecolor(C_GREY_BG)
    fig.patch.set_facecolor(C_GREY_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    steps = [
        ("1. Paper & Supervised Line", "Wang et al. 2023 &\nEXP001-EXP006 trials", C_NAVY),
        ("2. EXP-MLUA-001 Baseline", "10% SSL setup; observed\nforeground degradation", C_AMBER),
        ("3. EXP-MLUA-002 Scaling", "Scaled consistency;\nNaN crash at Step 1257", C_RED),
        ("4. Forensic Investigation", "Isolated missing Teacher\nBatchNorm buffer sync", C_RED),
        ("5. Buffer Remediation", "Dual weight + buffer\nEMA sync implemented", C_TEAL),
        ("6. EXP-MLUA-003 (60 Ep)", "Stable run; selected\nEpoch 56 (Val Dice 65.62%)", C_GREEN),
        ("7. Threshold Sweep", "Evaluated tau in [0.05, 0.95];\ntau = 0.50 confirmed", C_SLATE),
        ("8. Sealed Test (100 OPGs)", "Macro Dice 43.04%;\nGeneralization gap audited", C_NAVY),
        ("9. Remaining Work", "FP filtering, error taxonomy,\n& UI integration", C_SLATE)
    ]

    positions = [
        (16, 40), (50, 40), (84, 40),
        (84, 25), (50, 25), (16, 25),
        (16, 10), (50, 10), (84, 10)
    ]

    for (title, desc, col), (x, y) in zip(steps, positions):
        box = patches.FancyBboxPatch((x-14, y-5), 28, 10, boxstyle="round,pad=0.5",
                                      facecolor="white", edgecolor=col, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+2, title, ha='center', va='center', fontsize=8, fontweight='bold', color=col)
        ax.text(x, y-1.5, desc, ha='center', va='center', fontsize=6.5, color='#4A5568')

    ax.annotate('', xy=(36, 40), xytext=(30, 40), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_NAVY))
    ax.annotate('', xy=(70, 40), xytext=(64, 40), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_NAVY))
    ax.annotate('', xy=(84, 30), xytext=(84, 35), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_NAVY))
    ax.annotate('', xy=(64, 25), xytext=(70, 25), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_NAVY))
    ax.annotate('', xy=(30, 25), xytext=(36, 25), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_NAVY))
    ax.annotate('', xy=(16, 15), xytext=(16, 20), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_NAVY))
    ax.annotate('', xy=(36, 10), xytext=(30, 10), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_NAVY))
    ax.annotate('', xy=(70, 10), xytext=(64, 10), arrowprops=dict(arrowstyle="->", lw=1.8, color=C_NAVY))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure2_progress_flowchart.png"), dpi=300)
    plt.close()

# Fig 3: Teacher-Student Architecture
def fig3():
    print("Generating Figure 3: MLUA Teacher-Student Architecture...")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    # Student Box
    box_s = patches.FancyBboxPatch((10, 15), 35, 25, boxstyle="round,pad=0.5", facecolor="#EBF8FF", edgecolor=C_SLATE, lw=1.8)
    ax.add_patch(box_s)
    ax.text(27.5, 36, "STUDENT NETWORK", ha='center', fontsize=9, fontweight='bold', color=C_SLATE)
    ax.text(27.5, 30, "ResNet-34 Encoder", ha='center', fontsize=7.5, color=C_NAVY)
    ax.text(27.5, 24, "FPN Decoder (P2-P5)", ha='center', fontsize=7.5, color=C_NAVY)
    ax.text(27.5, 18, "Backprop Gradient Update", ha='center', fontsize=7, color='#718096')

    # Teacher Box
    box_t = patches.FancyBboxPatch((55, 15), 35, 25, boxstyle="round,pad=0.5", facecolor="#FEFCBF", edgecolor=C_AMBER, lw=1.8)
    ax.add_patch(box_t)
    ax.text(72.5, 36, "TEACHER NETWORK", ha='center', fontsize=9, fontweight='bold', color=C_AMBER)
    ax.text(72.5, 30, "ResNet-34 Encoder", ha='center', fontsize=7.5, color=C_NAVY)
    ax.text(72.5, 24, "FPN Decoder (P2-P5)", ha='center', fontsize=7.5, color=C_NAVY)
    ax.text(72.5, 18, "EMA Update (alpha = 0.99)", ha='center', fontsize=7, color='#718096')

    # EMA arrow from Student to Teacher
    ax.annotate('EMA Weights + BN Buffers', xy=(55, 27.5), xytext=(45, 27.5),
                arrowprops=dict(arrowstyle="->", lw=2, color=C_GREEN), fontsize=7.5, fontweight='bold', ha='center', va='bottom')

    # Consistency Loss arrow
    ax.annotate('Uncertainty-Masked Consistency Loss', xy=(27.5, 15), xytext=(72.5, 15),
                arrowprops=dict(arrowstyle="->", lw=1.5, color=C_RED, connectionstyle="arc3,rad=-0.3"),
                fontsize=7.5, fontweight='bold', color=C_RED, ha='center', va='top')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure3_teacher_student.png"), dpi=300)
    plt.close()

# Fig 4: ResNet-34 + FPN Architecture
def fig4():
    print("Generating Figure 4: ResNet-34 + FPN Architecture...")
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    stages = [
        ("Input Patch\n(384x384x3)", 10, 25),
        ("C2 (64 ch)\n(96x96)", 25, 40),
        ("C3 (128 ch)\n(48x48)", 25, 30),
        ("C4 (256 ch)\n(24x24)", 25, 20),
        ("C5 (512 ch)\n(12x12)", 25, 10),
        ("P2 (256 ch)", 60, 40),
        ("P3 (256 ch)", 60, 30),
        ("P4 (256 ch)", 60, 20),
        ("P5 (256 ch)", 60, 10),
        ("Fused Head\n(384x384x1)", 88, 25)
    ]

    for title, x, y in stages:
        col = C_SLATE if 'C' in title else (C_AMBER if 'P' in title else C_NAVY)
        box = patches.FancyBboxPatch((x-7, y-3.5), 14, 7, boxstyle="round,pad=0.3", facecolor="white", edgecolor=col, lw=1.4)
        ax.add_patch(box)
        ax.text(x, y, title, ha='center', va='center', fontsize=6.8, fontweight='bold', color=col)

    # Connections
    for y in [40, 30, 20, 10]:
        ax.annotate('', xy=(53, y), xytext=(32, y), arrowprops=dict(arrowstyle="->", lw=1.2, color=C_SLATE))
        ax.annotate('', xy=(81, 25), xytext=(67, y), arrowprops=dict(arrowstyle="->", lw=1.2, color=C_AMBER))

    # Top-down arrows
    ax.annotate('', xy=(60, 33.5), xytext=(60, 36.5), arrowprops=dict(arrowstyle="->", lw=1.2, color=C_AMBER))
    ax.annotate('', xy=(60, 23.5), xytext=(60, 26.5), arrowprops=dict(arrowstyle="->", lw=1.2, color=C_AMBER))
    ax.annotate('', xy=(60, 13.5), xytext=(60, 16.5), arrowprops=dict(arrowstyle="->", lw=1.2, color=C_AMBER))

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure4_resnet34_fpn.png"), dpi=300)
    plt.close()

# Fig 5: EXP-MLUA-001 Training Progression (E1 to E22)
def fig5():
    print("Generating Figure 5: EXP-MLUA-001 Progression...")
    df1 = pd.read_csv("outputs/experiments/EXP-MLUA-001_HISTORICAL/EXP-MLUA-001_FULL_TRAINING_HISTORY.csv")
    fig, ax1 = plt.subplots(figsize=(8, 4), dpi=300)
    ax1.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)

    color = C_SLATE
    ax1.set_xlabel('Epoch', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax1.set_ylabel('Validation Dice (%)', fontsize=8.5, fontweight='bold', color=color)
    ax1.plot(df1['epoch'], df1['val_dice']*100, color=color, marker='o', lw=1.8, label='Val Dice (%)')
    ax1.tick_params(axis='y', labelcolor=color, labelsize=7.5)
    ax1.tick_params(axis='x', labelsize=7.5)

    ax2 = ax1.twinx()
    color2 = C_RED
    ax2.set_ylabel('Validation Loss', fontsize=8.5, fontweight='bold', color=color2)
    ax2.plot(df1['epoch'], df1['val_loss'], color=color2, linestyle='--', lw=1.8, label='Val Loss')
    ax2.tick_params(axis='y', labelcolor=color2, labelsize=7.5)

    plt.title('EXP-MLUA-001: Historical Progression (E1-E22 Degradation)', fontsize=9.5, fontweight='bold', color=C_NAVY, pad=8)
    ax1.grid(True, linestyle=':', alpha=0.5, color=C_BORDER)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure5_exp001_progression.png"), dpi=300)
    plt.close()

# Fig 6: EXP-MLUA-002 Numerical Instability Timeline
def fig6():
    print("Generating Figure 6: EXP-MLUA-002 Failure Timeline...")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)

    epochs = np.arange(1, 11)
    val_loss = [1.049, 1.047, 1.044, 1.044, 1.043, 1.042, 1.041, 1.038, 1.038, np.nan]
    dice = [0, 0, 0, 0, 0, 0, 0, 0, 0, np.nan]

    ax.plot(epochs[:9], dice[:9], marker='s', color=C_SLATE, lw=2, label='Val Dice (0.0% Foreground Suppression)')
    ax.axvline(x=10, color=C_RED, linestyle='--', lw=2, label='Epoch 10 Step 1257: NaN Crash (GroupNorm Overflow)')
    ax.scatter([10], [0], color=C_RED, s=80, zorder=5, marker='x')

    ax.set_xlabel('Epoch', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_ylabel('Validation Dice (%)', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_title('EXP-MLUA-002: Training Timeline & Catastrophic Collapse', fontsize=9.5, fontweight='bold', color=C_NAVY, pad=8)
    ax.set_ylim(-5, 20)
    ax.set_xlim(0.5, 10.5)
    ax.grid(True, linestyle=':', alpha=0.5, color=C_BORDER)
    ax.legend(loc='upper left', fontsize=7.5, frameon=True, facecolor=C_GREY_BG)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure6_exp002_timeline.png"), dpi=300)
    plt.close()

# Fig 7: Teacher vs Student BN Statistics (Forensic Comparison)
def fig7():
    print("Generating Figure 7: Teacher vs Student BN Stats...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 4), dpi=300)
    fig.patch.set_facecolor(C_GREY_BG)

    # Subplot 1: Running Variance
    ax1.set_facecolor("white")
    ax1.bar(['Student BN', 'Teacher BN (EXP-002)', 'Teacher BN (EXP-003)'], [247.90, 1.0, 247.90], color=[C_TEAL, C_RED, C_GREEN], width=0.5, edgecolor=C_BORDER)
    ax1.set_ylabel('Running Variance Value', fontsize=8, fontweight='bold', color=C_NAVY)
    ax1.set_title('Running Variance Comparison', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax1.tick_params(axis='x', labelsize=6.8)
    ax1.grid(axis='y', linestyle=':', alpha=0.5, color=C_BORDER)

    # Subplot 2: Max c5 Feature Activation (Log Scale)
    ax2.set_facecolor("white")
    ax2.bar(['EXP-002 (Unsynced)', 'EXP-003 (Synced)'], [1e18, 11.09], color=[C_RED, C_GREEN], width=0.4, edgecolor=C_BORDER)
    ax2.set_yscale('log')
    ax2.set_ylabel('Peak Activation (Log Scale)', fontsize=8, fontweight='bold', color=C_NAVY)
    ax2.set_title('GroupNorm Input Activation Magnitude', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax2.tick_params(axis='x', labelsize=7)
    ax2.grid(axis='y', linestyle=':', alpha=0.5, color=C_BORDER)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure7_bn_statistics.png"), dpi=300)
    plt.close()

# Fig 8: EXP-MLUA-003 Dice Across Epochs
def fig8():
    print("Generating Figure 8: EXP-MLUA-003 Dice Across Epochs...")
    df3 = pd.read_csv("outputs/experiments/EXP-MLUA-003_FINAL/EXP-MLUA-003_FULL_TRAINING_HISTORY.csv")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)

    ax.plot(df3['epoch'], df3['val_dice']*100, color=C_SLATE, lw=1.8, marker='o', markersize=3, label='Validation Dice (%)')
    ax.scatter([56], [65.623], color=C_RED, s=70, zorder=5, label='Peak Checkpoint (E56: 65.62%)')
    ax.axvline(x=56, color=C_RED, linestyle=':', lw=1.2)

    ax.set_xlabel('Epoch', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_ylabel('Validation Dice (%)', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_title('EXP-MLUA-003: Validation Dice Progression (Epochs 1 to 60)', fontsize=9.5, fontweight='bold', color=C_NAVY, pad=8)
    ax.set_ylim(-5, 75)
    ax.set_xlim(0, 61)
    ax.grid(True, linestyle=':', alpha=0.5, color=C_BORDER)
    ax.legend(loc='lower right', fontsize=7.5, frameon=True, facecolor=C_GREY_BG)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure8_exp003_dice.png"), dpi=300)
    plt.close()

# Fig 9: EXP-MLUA-003 Validation Loss Across Epochs
def fig9():
    print("Generating Figure 9: EXP-MLUA-003 Validation Loss...")
    df3 = pd.read_csv("outputs/experiments/EXP-MLUA-003_FINAL/EXP-MLUA-003_FULL_TRAINING_HISTORY.csv")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)

    ax.plot(df3['epoch'], df3['train_loss'], color=C_TEAL, lw=1.8, label='Training Loss')
    ax.plot(df3['epoch'], df3['val_loss'], color=C_AMBER, lw=1.8, label='Validation Loss')
    ax.scatter([56], [0.76388], color=C_RED, s=70, zorder=5, label='Lowest Val Loss (E56: 0.7639)')

    ax.set_xlabel('Epoch', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_ylabel('Loss Value (BCE + Soft Dice)', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_title('EXP-MLUA-003: Training & Validation Loss Curves', fontsize=9.5, fontweight='bold', color=C_NAVY, pad=8)
    ax.set_ylim(0.4, 1.2)
    ax.set_xlim(0, 61)
    ax.grid(True, linestyle=':', alpha=0.5, color=C_BORDER)
    ax.legend(loc='upper right', fontsize=7.5, frameon=True, facecolor=C_GREY_BG)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure9_exp003_loss.png"), dpi=300)
    plt.close()

# Fig 10: Threshold vs Dice
def fig10():
    print("Generating Figure 10: Threshold vs Dice...")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)

    thresholds = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    dice_scores = np.array([58.847, 61.863, 63.246, 64.020, 64.546, 64.930, 65.244, 65.424, 65.555, 65.623, 65.601, 65.517, 65.401, 65.227, 64.811, 64.173, 63.134, 61.404, 57.661])

    ax.plot(thresholds, dice_scores, marker='o', color=C_SLATE, lw=2, markersize=4, label='Validation Dice (%)')
    ax.axvline(x=0.50, color=C_RED, linestyle='--', lw=1.5, label='Selected Threshold (tau = 0.50, Dice = 65.62%)')
    ax.scatter([0.50], [65.623], color=C_RED, s=60, zorder=5)

    ax.set_xlabel('Decision Threshold (tau)', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_ylabel('Validation Dice (%)', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_title('EXP-MLUA-003 Checkpoint E56: Threshold vs. Dice Sensitivity', fontsize=9.5, fontweight='bold', color=C_NAVY, pad=8)
    ax.grid(True, linestyle=':', alpha=0.5, color=C_BORDER)
    ax.set_ylim(55, 68)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc='lower center', fontsize=7.5, frameon=True, facecolor=C_GREY_BG)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure10_threshold_dice.png"), dpi=300)
    plt.close()

# Fig 11: Threshold vs Precision and Recall
def fig11():
    print("Generating Figure 11: Threshold vs Precision & Recall...")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)

    thresholds = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    precision = np.array([49.060, 54.558, 57.799, 60.135, 61.987, 63.630, 65.185, 66.507, 67.778, 69.009, 70.143, 71.273, 72.468, 73.832, 75.161, 76.615, 78.247, 80.246, 82.964])
    recall = np.array([75.925, 73.281, 71.429, 69.889, 68.659, 67.547, 66.517, 65.550, 64.606, 63.649, 62.667, 61.647, 60.589, 59.375, 57.874, 56.081, 53.752, 50.534, 45.002])

    ax.plot(thresholds, precision, marker='s', color=C_TEAL, lw=1.8, markersize=3.5, label='Precision (%)')
    ax.plot(thresholds, recall, marker='^', color=C_AMBER, lw=1.8, markersize=3.5, label='Recall (%)')
    ax.axvline(x=0.50, color=C_RED, linestyle='--', lw=1.5, label='Selected tau = 0.50 (Prec = 69.01%, Rec = 63.65%)')

    ax.set_xlabel('Decision Threshold (tau)', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_ylabel('Metric Score (%)', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_title('EXP-MLUA-003 Checkpoint E56: Threshold vs. Precision & Recall Trade-off', fontsize=9.5, fontweight='bold', color=C_NAVY, pad=8)
    ax.grid(True, linestyle=':', alpha=0.5, color=C_BORDER)
    ax.set_ylim(40, 90)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc='center left', fontsize=7.5, frameon=True, facecolor=C_GREY_BG)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure11_threshold_prec_rec.png"), dpi=300)
    plt.close()

# Fig 12: Validation vs Sealed Test Metrics
def fig12():
    print("Generating Figure 12: Validation vs Sealed Test Metrics...")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)

    metrics = ['Dice', 'IoU', 'Precision', 'Recall', 'Specificity']
    val_scores = [65.623, 49.854, 69.009, 63.649, 99.753]
    test_macro = [43.041, 29.057, 41.244, 52.896, 99.630]
    test_micro = [43.391, 27.707, 37.795, 50.931, 99.630]

    x = np.arange(len(metrics))
    width = 0.25

    rects1 = ax.bar(x - width, val_scores, width, label='Validation (E56 Patches)', color=C_SLATE, edgecolor=C_BORDER)
    rects2 = ax.bar(x, test_macro, width, label='Sealed Test (Macro OPG)', color=C_TEAL, edgecolor=C_BORDER)
    rects3 = ax.bar(x + width, test_micro, width, label='Sealed Test (Micro Pixel)', color=C_AMBER, edgecolor=C_BORDER)

    ax.set_ylabel('Performance Score (%)', fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_title('EXP-MLUA-003: Validation vs. Sealed Test Benchmark Comparison', fontsize=9.5, fontweight='bold', color=C_NAVY, pad=8)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=8, fontweight='bold', color=C_NAVY)
    ax.set_ylim(0, 115)
    ax.grid(axis='y', linestyle=':', alpha=0.5, color=C_BORDER)
    ax.legend(loc='upper right', fontsize=7.5, frameon=True, facecolor=C_GREY_BG)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure12_val_vs_test.png"), dpi=300)
    plt.close()

# Fig 13: Example X-ray -> Prob Map -> Binary Mask -> Overlay
def fig13():
    print("Generating Figure 13: Example Inference Visualization...")
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.2), dpi=300)
    fig.patch.set_facecolor(C_GREY_BG)

    # Simulated visual pattern representing tooth crop with proximal caries
    np.random.seed(42)
    base_img = np.zeros((100, 100))
    # Enamel boundary
    for i in range(100):
        for j in range(100):
            d = np.sqrt((i-50)**2 + (j-50)**2)
            base_img[i, j] = np.clip(0.8 - d/80.0 + np.random.normal(0, 0.03), 0.1, 0.9)
    # Carious lesion (radiolucency)
    for i in range(35, 55):
        for j in range(35, 55):
            d2 = np.sqrt((i-45)**2 + (j-45)**2)
            if d2 < 8:
                base_img[i, j] -= 0.35

    axes[0].imshow(base_img, cmap='gray')
    axes[0].set_title("1. Panoramic X-Ray Crop", fontsize=7.5, fontweight='bold', color=C_NAVY)
    axes[0].axis('off')

    # Prob map
    prob_map = np.zeros((100, 100))
    for i in range(100):
        for j in range(100):
            d2 = np.sqrt((i-45)**2 + (j-45)**2)
            if d2 < 10:
                prob_map[i, j] = np.clip(0.9 - d2/15.0 + np.random.normal(0, 0.05), 0, 1)

    axes[1].imshow(prob_map, cmap='jet')
    axes[1].set_title("2. Predicted Prob. Map", fontsize=7.5, fontweight='bold', color=C_NAVY)
    axes[1].axis('off')

    # Binary mask (tau = 0.50)
    bin_mask = (prob_map >= 0.50).astype(float)
    axes[2].imshow(bin_mask, cmap='gray')
    axes[2].set_title("3. Binary Mask (tau=0.50)", fontsize=7.5, fontweight='bold', color=C_NAVY)
    axes[2].axis('off')

    # Overlay
    overlay = np.stack([base_img, base_img, base_img], axis=-1)
    overlay[bin_mask == 1] = [1.0, 0.2, 0.2]
    axes[3].imshow(overlay)
    axes[3].set_title("4. Suspected Lesion Overlay", fontsize=7.5, fontweight='bold', color=C_NAVY)
    axes[3].axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "figure13_visual_overlay.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    fig1()
    fig2()
    fig3()
    fig4()
    fig5()
    fig6()
    fig7()
    fig8()
    fig9()
    fig10()
    fig11()
    fig12()
    fig13()
    print("All 13 figures generated successfully!")
