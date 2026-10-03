import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs(os.path.join("outputs", "progress_figures"), exist_ok=True)

# Set global styles
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['figure.dpi'] = 300

# Palette
C_NAVY = "#1A365D"
C_SLATE = "#2B6CB0"
C_TEAL = "#2C7A7B"
C_AMBER = "#D69E2E"
C_RED = "#E53E3E"
C_GREEN = "#38A169"
C_GREY_BG = "#F7FAFC"
C_BOX_BG = "#EDF2F7"
C_BORDER = "#CBD5E0"

# -------------------------------------------------------------
# Figure 1: Project Progress Flowchart (Chronological Journey)
# -------------------------------------------------------------
def gen_fig1():
    print("Generating Progress Figure 1: Progress Journey Flowchart...")
    fig, ax = plt.subplots(figsize=(10, 5), dpi=300)
    ax.set_facecolor(C_GREY_BG)
    fig.patch.set_facecolor(C_GREY_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 50)
    ax.axis('off')

    steps = [
        ("1. MLUA Paper & DC1000", "Study Wang et al. 2023 &\nDC1000 dataset (1000 OPGs)", C_NAVY),
        ("2. Initial Implementation", "PyTorch ResNet-34 + FPN\nTeacher-Student setup", C_SLATE),
        ("3. EXP-MLUA-001", "Baseline training run;\nobserved degradation", C_AMBER),
        ("4. EXP-MLUA-002", "Training collapsed;\nNaN loss at Teacher GroupNorm", C_RED),
        ("5. Forensic Investigation", "Isolated missing Teacher\nBatchNorm buffer sync", C_RED),
        ("6. Buffer Remediation", "Dual weight + buffer\nEMA sync implemented", C_TEAL),
        ("7. EXP-MLUA-003", "60 epochs completed;\nE56 Val Dice = 65.62%", C_GREEN),
        ("8. Threshold & Sealed Test", "tau=0.50 selected;\nMacro Dice = 43.04%", C_NAVY),
        ("9. Remaining Work", "Generalization, FP reduction,\n& app integration", C_SLATE)
    ]

    # Draw 3 rows of 3
    positions = [
        (16, 40), (50, 40), (84, 40),
        (84, 25), (50, 25), (16, 25),
        (16, 10), (50, 10), (84, 10)
    ]

    for i, ((title, desc, col), (x, y)) in enumerate(zip(steps, positions)):
        box = patches.FancyBboxPatch((x-14, y-5), 28, 10, boxstyle="round,pad=0.5",
                                      facecolor="white", edgecolor=col, linewidth=2)
        ax.add_patch(box)
        ax.text(x, y+2, title, ha='center', va='center', fontsize=8.5, fontweight='bold', color=col)
        ax.text(x, y-1.5, desc, ha='center', va='center', fontsize=6.5, color='#4A5568')

    # Draw arrows
    # Row 1: 0->1, 1->2
    ax.annotate('', xy=(36, 40), xytext=(30, 40), arrowprops=dict(arrowstyle="->", lw=2, color=C_NAVY))
    ax.annotate('', xy=(70, 40), xytext=(64, 40), arrowprops=dict(arrowstyle="->", lw=2, color=C_NAVY))
    # Down 2->3
    ax.annotate('', xy=(84, 30), xytext=(84, 35), arrowprops=dict(arrowstyle="->", lw=2, color=C_NAVY))
    # Row 2: 3->4, 4->5
    ax.annotate('', xy=(64, 25), xytext=(70, 25), arrowprops=dict(arrowstyle="->", lw=2, color=C_NAVY))
    ax.annotate('', xy=(30, 25), xytext=(36, 25), arrowprops=dict(arrowstyle="->", lw=2, color=C_NAVY))
    # Down 5->6
    ax.annotate('', xy=(16, 15), xytext=(16, 20), arrowprops=dict(arrowstyle="->", lw=2, color=C_NAVY))
    # Row 3: 6->7, 7->8
    ax.annotate('', xy=(36, 10), xytext=(30, 10), arrowprops=dict(arrowstyle="->", lw=2, color=C_NAVY))
    ax.annotate('', xy=(70, 10), xytext=(64, 10), arrowprops=dict(arrowstyle="->", lw=2, color=C_NAVY))

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "progress_figures", "fig1_progress_flowchart.png"), dpi=300)
    plt.close()

# -------------------------------------------------------------
# Figure 2: System Architecture & Data Flowchart
# -------------------------------------------------------------
def gen_fig2():
    print("Generating Progress Figure 2: System Architecture Flowchart...")
    fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
    ax.set_facecolor(C_GREY_BG)
    fig.patch.set_facecolor(C_GREY_BG)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 60)
    ax.axis('off')

    # Top: Dataset
    box1 = patches.FancyBboxPatch((35, 50), 30, 7, boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_NAVY, lw=1.5)
    ax.add_patch(box1)
    ax.text(50, 54.5, "DC1000 Panoramic X-ray Dataset", ha='center', va='center', fontsize=9, fontweight='bold', color=C_NAVY)
    ax.text(50, 52, "1,000 OPGs -> 768x1536 Standardized Res.", ha='center', va='center', fontsize=7, color='#4A5568')

    # Patches
    box2 = patches.FancyBboxPatch((35, 39), 30, 7, boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_SLATE, lw=1.5)
    ax.add_patch(box2)
    ax.text(50, 43.5, "384x384 Patch Extraction (Stride=192)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_SLATE)
    ax.text(50, 41, "20% Labeled (530) / 80% Unlabeled (1,859)", ha='center', va='center', fontsize=7, color='#4A5568')

    ax.annotate('', xy=(50, 46), xytext=(50, 50), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_NAVY))

    # Split: Student & Teacher
    # Left: Student
    box_s = patches.FancyBboxPatch((8, 24), 38, 11, boxstyle="round,pad=0.4", facecolor="white", edgecolor=C_TEAL, lw=1.5)
    ax.add_patch(box_s)
    ax.text(27, 32, "STUDENT NETWORK", ha='center', va='center', fontsize=9, fontweight='bold', color=C_TEAL)
    ax.text(27, 29, "ResNet-34 Encoder + FPN Decoder", ha='center', va='center', fontsize=7.5, color='#2D3748')
    ax.text(27, 26.5, "Supervised Loss (BCE + Soft Dice) on Labeled", ha='center', va='center', fontsize=6.5, color='#4A5568')

    # Right: Teacher
    box_t = patches.FancyBboxPatch((54, 24), 38, 11, boxstyle="round,pad=0.4", facecolor="white", edgecolor=C_AMBER, lw=1.5)
    ax.add_patch(box_t)
    ax.text(73, 32, "TEACHER NETWORK (EMA Updated)", ha='center', va='center', fontsize=9, fontweight='bold', color=C_AMBER)
    ax.text(73, 29, "ResNet-34 Encoder + FPN Decoder", ha='center', va='center', fontsize=7.5, color='#2D3748')
    ax.text(73, 26.5, "Monte Carlo (T=8) -> Uncertainty & Conf. Mask", ha='center', va='center', fontsize=6.5, color='#4A5568')

    ax.annotate('', xy=(27, 35), xytext=(45, 39), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_NAVY))
    ax.annotate('', xy=(73, 35), xytext=(55, 39), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_NAVY))

    # Bottom Center: Consistency & Optimization
    box_opt = patches.FancyBboxPatch((20, 11), 60, 9, boxstyle="round,pad=0.4", facecolor="white", edgecolor=C_NAVY, lw=1.5)
    ax.add_patch(box_opt)
    ax.text(50, 16.5, "Multi-Task Optimization & Remediation", ha='center', va='center', fontsize=9, fontweight='bold', color=C_NAVY)
    ax.text(50, 13.5, "L_total = L_sup + lambda * L_con (Confidence Masked) | Dual Weight + BN Buffer EMA Sync", ha='center', va='center', fontsize=6.8, color='#2D3748')

    ax.annotate('', xy=(38, 20), xytext=(27, 24), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_TEAL))
    ax.annotate('', xy=(62, 20), xytext=(73, 24), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_AMBER))

    # Final Output Box
    box_out = patches.FancyBboxPatch((25, 1), 50, 7, boxstyle="round,pad=0.3", facecolor="white", edgecolor=C_GREEN, lw=1.5)
    ax.add_patch(box_out)
    ax.text(50, 5.5, "Sliding-Window Full OPG Reconstruction (tau = 0.50)", ha='center', va='center', fontsize=8.5, fontweight='bold', color=C_GREEN)
    ax.text(50, 3, "Validation Dice: 65.62% | Sealed Test Macro Dice: 43.04%", ha='center', va='center', fontsize=7, color='#4A5568')

    ax.annotate('', xy=(50, 8), xytext=(50, 11), arrowprops=dict(arrowstyle="->", lw=1.5, color=C_NAVY))

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "progress_figures", "fig2_system_flowchart.png"), dpi=300)
    plt.close()

# -------------------------------------------------------------
# Figure 3: EXP-MLUA-002 Failure vs EXP-MLUA-003 Remediation
# -------------------------------------------------------------
def gen_fig3():
    print("Generating Progress Figure 3: Failure vs Remediation...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5), dpi=300)
    fig.patch.set_facecolor(C_GREY_BG)

    # Subplot 1: EXP-002
    ax1.set_facecolor("white")
    ax1.set_title("EXP-MLUA-002: Missing Buffer Sync", fontsize=10, fontweight='bold', color=C_RED, pad=10)
    stages = ["Student\nWeights", "Teacher\nWeights (EMA)", "Student\nBN Buffers", "Teacher\nBN Buffers", "Teacher\nGroupNorm", "Training\nState"]
    vals = [1.0, 1.0, 1.0, 0.0, 10.0, 0.0]
    colors_list = [C_TEAL, C_AMBER, C_TEAL, C_RED, C_RED, C_RED]
    bars = ax1.bar(stages, [1, 1, 1, 0.05, 1, 0.05], color=colors_list, width=0.5, edgecolor=C_BORDER)
    ax1.set_ylim(0, 1.3)
    ax1.set_yticks([])
    ax1.text(0, 1.05, "Updated", ha='center', fontsize=7, color=C_TEAL, fontweight='bold')
    ax1.text(1, 1.05, "EMA Sync", ha='center', fontsize=7, color=C_AMBER, fontweight='bold')
    ax1.text(2, 1.05, "Tracking", ha='center', fontsize=7, color=C_TEAL, fontweight='bold')
    ax1.text(3, 0.1, "STALE/FROZEN", ha='center', fontsize=6.5, color=C_RED, fontweight='bold')
    ax1.text(4, 1.05, "NaN / INF", ha='center', fontsize=7, color=C_RED, fontweight='bold')
    ax1.text(5, 0.1, "COLLAPSED", ha='center', fontsize=6.5, color=C_RED, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=7)

    # Subplot 2: EXP-003
    ax2.set_facecolor("white")
    ax2.set_title("EXP-MLUA-003: Corrected Dual Sync", fontsize=10, fontweight='bold', color=C_GREEN, pad=10)
    bars2 = ax2.bar(stages, [1, 1, 1, 1, 1, 1], color=[C_TEAL, C_AMBER, C_TEAL, C_GREEN, C_GREEN, C_GREEN], width=0.5, edgecolor=C_BORDER)
    ax2.set_ylim(0, 1.3)
    ax2.set_yticks([])
    ax2.text(0, 1.05, "Updated", ha='center', fontsize=7, color=C_TEAL, fontweight='bold')
    ax2.text(1, 1.05, "EMA Sync", ha='center', fontsize=7, color=C_AMBER, fontweight='bold')
    ax2.text(2, 1.05, "Tracking", ha='center', fontsize=7, color=C_TEAL, fontweight='bold')
    ax2.text(3, 1.05, "EMA SYNCED", ha='center', fontsize=7, color=C_GREEN, fontweight='bold')
    ax2.text(4, 1.05, "Stable (~1.2)", ha='center', fontsize=7, color=C_GREEN, fontweight='bold')
    ax2.text(5, 1.05, "60 EPOCHS OK", ha='center', fontsize=7, color=C_GREEN, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=7)

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "progress_figures", "fig3_failure_remediation.png"), dpi=300)
    plt.close()

# -------------------------------------------------------------
# Figure 4: Validation Threshold Sensitivity (tau 0.05 to 0.95)
# -------------------------------------------------------------
def gen_fig4():
    print("Generating Progress Figure 4: Threshold Sensitivity...")
    fig, ax = plt.subplots(figsize=(8, 4), dpi=300)
    ax.set_facecolor("white")
    fig.patch.set_facecolor(C_GREY_BG)

    thresholds = np.array([0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    dice_scores = np.array([58.847, 61.863, 63.246, 64.020, 64.546, 64.930, 65.244, 65.424, 65.555, 65.623, 65.601, 65.517, 65.401, 65.227, 64.811, 64.173, 63.134, 61.404, 57.661])

    ax.plot(thresholds, dice_scores, marker='o', color=C_SLATE, linewidth=2, markersize=4.5, label='Validation Dice (%)')
    ax.axvline(x=0.50, color=C_RED, linestyle='--', linewidth=1.5, label='Selected Threshold (tau = 0.50, Dice = 65.62%)')
    ax.scatter([0.50], [65.623], color=C_RED, s=60, zorder=5)

    ax.set_xlabel("Decision Threshold (tau)", fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_ylabel("Validation Dice Score (%)", fontsize=8.5, fontweight='bold', color=C_NAVY)
    ax.set_title("EXP-MLUA-003 Checkpoint E56: Validation Threshold Sensitivity", fontsize=9.5, fontweight='bold', color=C_NAVY, pad=8)
    ax.grid(True, linestyle=':', alpha=0.6, color=C_BORDER)
    ax.set_ylim(55, 68)
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc='lower center', fontsize=7.5, frameon=True, facecolor=C_GREY_BG)

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "progress_figures", "fig4_threshold_sensitivity.png"), dpi=300)
    plt.close()

# -------------------------------------------------------------
# Figure 5: Validation vs Sealed Test Performance
# -------------------------------------------------------------
def gen_fig5():
    print("Generating Progress Figure 5: Val vs Sealed Test Comparison...")
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
    ax.grid(axis='y', linestyle=':', alpha=0.6, color=C_BORDER)
    ax.legend(loc='upper right', fontsize=7.5, frameon=True, facecolor=C_GREY_BG)

    # Value labels
    for rect in rects1:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=6, color=C_SLATE, fontweight='bold')
    for rect in rects2:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=6, color=C_TEAL, fontweight='bold')
    for rect in rects3:
        h = rect.get_height()
        ax.annotate(f'{h:.1f}%', xy=(rect.get_x() + rect.get_width() / 2, h), xytext=(0, 2),
                    textcoords="offset points", ha='center', va='bottom', fontsize=6, color=C_AMBER, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join("outputs", "progress_figures", "fig5_val_vs_test.png"), dpi=300)
    plt.close()

if __name__ == "__main__":
    gen_fig1()
    gen_fig2()
    gen_fig3()
    gen_fig4()
    gen_fig5()
    print("All progress figures generated successfully!")
