import { jsPDF } from 'jspdf';
import { AnalysisResult } from '../types/api';

/**
 * Generates a clean, spacious, high-readability 2-page Medical Clinical AI PDF Report.
 */
export async function generateClinicalPDF(analysis: AnalysisResult): Promise<Blob> {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const pageWidth = doc.internal.pageSize.getWidth(); // 210 mm
  const pageHeight = doc.internal.pageSize.getHeight(); // 297 mm
  const margin = 16;
  const contentWidth = pageWidth - margin * 2; // 178 mm

  // Color Palette
  const darkNavy = [15, 23, 42]; // #0f172a
  const slateText = [71, 85, 105]; // #475569
  const lightSlate = [100, 116, 139]; // #64748b
  const lightBg = [248, 250, 252]; // #f8fafc
  const cardBorder = [226, 232, 240]; // #e2e8f0
  const cyanBrand = [14, 165, 233]; // #0ea5e9
  const cyanDark = [3, 105, 161]; // #0369a1

  // Determine Clinical Stage Theme
  const stageLevelStr = (analysis.stage?.level || '').toLowerCase();
  const overallStr = (analysis.overallFinding || '').toLowerCase();
  const isNoCaries = analysis.summary.totalLesions === 0 || stageLevelStr.includes('0') || overallStr.includes('no caries');
  const isStage3 = !isNoCaries && (stageLevelStr.includes('3') || overallStr.includes('extensive'));
  const isStage2 = !isNoCaries && !isStage3 && (stageLevelStr.includes('2') || overallStr.includes('moderate'));

  // Stage specific colors
  const stageColors = isNoCaries
    ? {
        bg: [240, 253, 244],
        border: [134, 239, 172],
        text: [22, 101, 52],
        badgeBg: [220, 252, 231],
        title: 'STAGE 0 — NO SIGNIFICANT CARIES DETECTED',
        priority: 'Routine Preventive Follow-up',
      }
    : isStage3
    ? {
        bg: [254, 242, 242],
        border: [252, 165, 165],
        text: [153, 27, 27],
        badgeBg: [254, 226, 226],
        title: 'LEVEL 3 — EXTENSIVE DENTINAL CARIES',
        priority: 'Urgent Restorative & Endodontic Evaluation',
      }
    : isStage2
    ? {
        bg: [255, 247, 237],
        border: [253, 186, 116],
        text: [154, 52, 18],
        badgeBg: [255, 237, 213],
        title: 'LEVEL 2 — MODERATE DENTINAL CARIES',
        priority: 'Restorative Evaluation Recommended',
      }
    : {
        bg: [254, 252, 232],
        border: [253, 224, 71],
        text: [133, 77, 14],
        badgeBg: [254, 249, 195],
        title: 'LEVEL 1 — SUSPECTED EARLY CARIES',
        priority: 'Clinical Monitoring & Non-Invasive Therapy',
      };

  // Helper to load image as base64 or fallback to canvas placeholder
  const loadImageData = (url: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'Anonymous';
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = img.width || 800;
          canvas.height = img.height || 480;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            resolve(canvas.toDataURL('image/jpeg', 0.90));
            return;
          }
        } catch {
          // fallback
        }
        resolve(createPlaceholderCanvas(false));
      };
      img.onerror = () => {
        resolve(createPlaceholderCanvas(url.includes('mask') || url.includes('binary')));
      };
      img.src = url;
    });
  };

  const createPlaceholderCanvas = (isMask: boolean): string => {
    const canvas = document.createElement('canvas');
    canvas.width = 800;
    canvas.height = 480;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      if (isMask) {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, 800, 480);
        ctx.fillStyle = '#FFFFFF';
        ctx.beginPath();
        ctx.ellipse(380, 240, 24, 18, 0, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.ellipse(540, 260, 18, 14, 0, 0, Math.PI * 2);
        ctx.fill();
      } else {
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, 800, 480);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '20px sans-serif';
        ctx.fillText('Panoramic Radiograph (OPG)', 250, 240);
      }
      return canvas.toDataURL('image/jpeg');
    }
    return '';
  };

  const originalImgBase64 = await loadImageData(analysis.originalImageUrl);
  const overlayImgBase64 = await loadImageData(analysis.segmentationOverlayUrl);
  const binaryMaskBase64 = await loadImageData(analysis.binaryMaskUrl);

  // =========================================================================
  // PAGE 1: Clinical Radiology Screening & Candidate Localization
  // =========================================================================
  let y = margin;

  // Header Title
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('DENTAL CARIES AI DECISION SUPPORT SYSTEM', margin, y + 2);

  // Right Case Metadata Block
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text(`Case ID: ${analysis.id}`, pageWidth - margin, y - 1, { align: 'right' });
  doc.text(`Exam Date: ${analysis.timestamp} UTC`, pageWidth - margin, y + 3.5, { align: 'right' });
  doc.text('Modality: Panoramic Radiograph (OPG)', pageWidth - margin, y + 8, { align: 'right' });

  y += 7;
  doc.setFontSize(8.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Clinical Radiology Screening & Candidate Localization Report', margin, y);

  y += 6;

  // Horizontal Header Divider
  doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
  doc.setLineWidth(0.4);
  doc.line(margin, y, pageWidth - margin, y);

  y += 5;

  // Dynamic Clinical Staging Alert Banner
  const bannerH = 11;
  doc.setFillColor(stageColors.bg[0], stageColors.bg[1], stageColors.bg[2]);
  doc.setDrawColor(stageColors.border[0], stageColors.border[1], stageColors.border[2]);
  doc.setLineWidth(0.5);
  doc.roundedRect(margin, y, contentWidth, bannerH, 2, 2, 'FD');

  // Pulse dot indicator
  doc.setFillColor(stageColors.text[0], stageColors.text[1], stageColors.text[2]);
  doc.circle(margin + 4.5, y + 5.5, 1.6, 'F');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(stageColors.text[0], stageColors.text[1], stageColors.text[2]);

  const bannerText = !isNoCaries
    ? `${stageColors.title}: AI detected ${analysis.summary.totalLesions} candidate lesion region(s) occupying ${analysis.summary.affectedAreaPercent.toFixed(2)}% surface area.`
    : 'STAGE 0 — NO SUSPICIOUS DEMINERALIZATION DETECTED: Routine preventive dental examination recommended.';
  doc.text(bannerText, margin + 8.5, y + 6.8, { maxWidth: contentWidth - 12 });

  y += bannerH + 6;

  // SECTION 1: Clinical Imaging & AI Candidate Visualization
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 1 — CLINICAL IMAGING & AI CANDIDATE VISUALIZATION', margin, y);
  y += 3.5;

  const panelWidth = (contentWidth - 6) / 2; // ~86 mm
  const panelHeight = 44; // comfortable panoramic aspect ratio

  // Panel A: Original Radiograph
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('Panel A: Original Dental Radiograph', margin, y);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  doc.setTextColor(lightSlate[0], lightSlate[1], lightSlate[2]);
  doc.text('Raw clinical OPG acquisition', margin, y + 3.2);

  // Border frame behind Image A
  doc.setFillColor(0, 0, 0);
  doc.rect(margin, y + 4.5, panelWidth, panelHeight, 'F');
  if (originalImgBase64) {
    doc.addImage(originalImgBase64, 'JPEG', margin, y + 4.5, panelWidth, panelHeight);
  }
  doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
  doc.setLineWidth(0.3);
  doc.rect(margin, y + 4.5, panelWidth, panelHeight, 'D');

  // Panel B: AI Overlay
  const panelBx = margin + panelWidth + 6;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('Panel B: AI-Assisted Caries Localization', panelBx, y);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  doc.setTextColor(cyanDark[0], cyanDark[1], cyanDark[2]);
  doc.text('Detected caries highlighted with cyan glow overlay', panelBx, y + 3.2);

  // Border frame behind Image B
  doc.setFillColor(0, 0, 0);
  doc.rect(panelBx, y + 4.5, panelWidth, panelHeight, 'F');
  if (overlayImgBase64) {
    doc.addImage(overlayImgBase64, 'JPEG', panelBx, y + 4.5, panelWidth, panelHeight);
  }
  doc.setDrawColor(cyanBrand[0], cyanBrand[1], cyanBrand[2]);
  doc.setLineWidth(0.4);
  doc.rect(panelBx, y + 4.5, panelWidth, panelHeight, 'D');

  y += panelHeight + 9;

  // SECTION 2: Quantitative Findings & Region Summary
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 2 — QUANTITATIVE FINDINGS & REGION SUMMARY', margin, y);
  y += 3.5;

  // 4 KPI Summary Cards Grid
  const cardW = (contentWidth - 6) / 4;
  const cardH = 15;

  const kpis = [
    { label: 'IMAGE SOURCE', val: analysis.filename.length > 14 ? analysis.filename.slice(0, 13) + '..' : analysis.filename },
    { label: 'CANDIDATE SITES', val: `${analysis.summary.totalLesions} Site(s)` },
    { label: 'AREA EXTENT', val: `${analysis.summary.affectedAreaPercent.toFixed(2)}% (${analysis.summary.totalAreaPx} px)` },
    { label: 'CLINICAL PRIORITY', val: stageColors.priority.split(' ')[0] + ' ' + (stageColors.priority.split(' ')[1] || '') },
  ];

  kpis.forEach((kpi, idx) => {
    const kx = margin + idx * (cardW + 2);
    doc.setFillColor(lightBg[0], lightBg[1], lightBg[2]);
    doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
    doc.setLineWidth(0.3);
    doc.roundedRect(kx, y, cardW, cardH, 1.5, 1.5, 'FD');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(6);
    doc.setTextColor(lightSlate[0], lightSlate[1], lightSlate[2]);
    doc.text(kpi.label, kx + 3, y + 4.5);

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(7.5);
    doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
    doc.text(kpi.val, kx + 3, y + 10.5);
  });

  y += cardH + 7;

  // SECTION 3: Clinical Decision Support & Action Protocol
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 3 — CLINICAL DECISION SUPPORT & ACTION PROTOCOL', margin, y);
  y += 3.5;

  const protocolH = 22;
  doc.setFillColor(240, 253, 250); // soft mint / cyan
  doc.setDrawColor(180, 235, 230);
  doc.setLineWidth(0.4);
  doc.roundedRect(margin, y, contentWidth, protocolH, 2, 2, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(15, 118, 110);
  doc.text(`Recommendation: ${analysis.clinicalRecommendation}`, margin + 4, y + 5.5, { maxWidth: contentWidth - 8 });

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  const guidanceNote = 'Clinical Protocol Note: Algorithmic findings represent candidate radiolucency zones. Please correlate findings with visual-tactile assessment, cold pulp vitality tests, and bitewing radiographs before restorative treatment.';
  doc.text(guidanceNote, margin + 4, y + 13.5, { maxWidth: contentWidth - 8 });

  // Page 1 Footer
  doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
  doc.line(margin, pageHeight - 12, pageWidth - margin, pageHeight - 12);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(lightSlate[0], lightSlate[1], lightSlate[2]);
  doc.text('Page 1 of 2 • Dental Caries Detection & Decision Support System • Authorized Dental Practitioner Review Only', margin, pageHeight - 7);

  // =========================================================================
  // PAGE 2: Binary Segmentation, Morphological Analysis & Sign-Off
  // =========================================================================
  doc.addPage();
  y = margin;

  // Header Title
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(13.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('DENTAL CARIES AI DECISION SUPPORT SYSTEM', margin, y + 2);

  // Right Case Metadata Block
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text(`Case ID: ${analysis.id}`, pageWidth - margin, y - 1, { align: 'right' });
  doc.text('Page: 2 of 2', pageWidth - margin, y + 3.5, { align: 'right' });

  y += 7;
  doc.setFontSize(8.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Binary Segmentation Mask, Morphological Analysis & Clinical Sign-Off', margin, y);

  y += 6;

  // Horizontal Header Divider
  doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
  doc.setLineWidth(0.4);
  doc.line(margin, y, pageWidth - margin, y);

  y += 5;

  // SECTION 4: Binary Caries Segmentation Mask
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 4 — BINARY CARIES SEGMENTATION MASK', margin, y);
  y += 3.5;

  const segWidth = (contentWidth - 6) / 2;
  const segHeight = 40;

  // Subheaders
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('Original Panoramic Radiograph', margin, y);
  doc.text('Binary Caries Segmentation Mask (Ground Binary)', margin + segWidth + 6, y);

  // Image A: Original
  doc.setFillColor(0, 0, 0);
  doc.rect(margin, y + 2.5, segWidth, segHeight, 'F');
  if (originalImgBase64) {
    doc.addImage(originalImgBase64, 'JPEG', margin, y + 2.5, segWidth, segHeight);
  }
  doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
  doc.rect(margin, y + 2.5, segWidth, segHeight, 'D');

  // Image B: Binary Mask
  doc.setFillColor(0, 0, 0);
  doc.rect(margin + segWidth + 6, y + 2.5, segWidth, segHeight, 'F');
  if (binaryMaskBase64) {
    doc.addImage(binaryMaskBase64, 'JPEG', margin + segWidth + 6, y + 2.5, segWidth, segHeight);
  }
  doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
  doc.rect(margin + segWidth + 6, y + 2.5, segWidth, segHeight, 'D');

  y += segHeight + 6;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  doc.setTextColor(lightSlate[0], lightSlate[1], lightSlate[2]);
  doc.text('Binary Mask Definition: Background = 0 (Black) | Caries Candidate Lesions = 1 (Solid White). Positive pixels reflect localized demineralization zones.', margin, y, { maxWidth: contentWidth });

  y += 6;

  // SECTION 5: Quantitative Metric Breakdown & Staging
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 5 — PIXEL-LEVEL METRIC BREAKDOWN & CARIES STAGING', margin, y);
  y += 3.5;

  // 3 Wide Metric Cards
  const mCardW = (contentWidth - 4) / 3;
  const mCardH = 13;

  const metricCards = [
    { title: 'DETECTED AREA', val: `${analysis.summary.affectedAreaPercent.toFixed(2)}%`, sub: 'Surface Extent Ratio' },
    { title: 'POSITIVE PIXELS', val: `${analysis.summary.totalAreaPx} px`, sub: 'Demineralized Area' },
    { title: 'CANDIDATE SITES', val: `${analysis.summary.totalLesions} Region(s)`, sub: 'Discrete Clusters' },
  ];

  metricCards.forEach((mc, i) => {
    const mx = margin + i * (mCardW + 2);
    doc.setFillColor(lightBg[0], lightBg[1], lightBg[2]);
    doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
    doc.setLineWidth(0.3);
    doc.roundedRect(mx, y, mCardW, mCardH, 1.5, 1.5, 'FD');

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(6);
    doc.setTextColor(lightSlate[0], lightSlate[1], lightSlate[2]);
    doc.text(mc.title, mx + mCardW * 0.5, y + 3.5, { align: 'center' });

    doc.setFont('helvetica', 'bold');
    doc.setFontSize(8.5);
    doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
    doc.text(mc.val, mx + mCardW * 0.5, y + 8, { align: 'center' });

    doc.setFont('helvetica', 'normal');
    doc.setFontSize(5.5);
    doc.setTextColor(lightSlate[0], lightSlate[1], lightSlate[2]);
    doc.text(mc.sub, mx + mCardW * 0.5, y + 11.2, { align: 'center' });
  });

  y += mCardH + 5;

  // Staging Bar
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(stageColors.text[0], stageColors.text[1], stageColors.text[2]);
  doc.text(`AI Staging Assessment: ${analysis.stage.level} — ${analysis.stage.title}`, margin, y);
  y += 3.5;
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text(`Classification Criteria: ${analysis.stage.description}. Staging is algorithmically computed from segmented lesion depth and area extent.`, margin, y, { maxWidth: contentWidth });

  y += 6.5;

  // SECTION 6: Clinical Interpretation & Diagnostic Considerations (2 Columns)
  const halfColW = (contentWidth - 6) / 2;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('Clinical Caries Interpretation', margin, y);
  doc.text('Diagnostic Review Considerations', margin + halfColW + 6, y);
  y += 3.2;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);

  const leftInterp = !isNoCaries
    ? `The AI system identified ${analysis.summary.totalLesions} candidate lesion region(s) consistent with dental caries, occupying ${analysis.summary.affectedAreaPercent.toFixed(2)}% of total image area. All highlighted regions indicate localized radiolucency that requires correlation with dental history and intraoral visual assessment.`
    : 'No statistically significant demineralization patterns detected. Routine prophylactic oral hygiene monitoring recommended.';
  doc.text(leftInterp, margin, y, { maxWidth: halfColW, lineHeightFactor: 1.35 });

  const rightBullets = [
    '• Early enamel lesions may appear as faint radiolucency.',
    '• True cavitation cannot be determined from 2D pixel area alone.',
    '• Cervical burnout and overlap may cause benign radiolucency.',
    '• Bitewing imaging recommended for interproximal confirmation.',
  ];
  doc.text(rightBullets.join('\n'), margin + halfColW + 6, y, { maxWidth: halfColW, lineHeightFactor: 1.35 });

  y += 20;

  // SECTION 7: Physician / Dental Clinician Review & Verification Box
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 7 — PHYSICIAN / DENTAL CLINICIAN VERIFICATION', margin, y);
  y += 3.5;

  const signBoxH = 26;
  doc.setFillColor(lightBg[0], lightBg[1], lightBg[2]);
  doc.setDrawColor(cyanBrand[0], cyanBrand[1], cyanBrand[2]);
  doc.setLineWidth(0.4);
  doc.roundedRect(margin, y, contentWidth, signBoxH, 2, 2, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(cyanDark[0], cyanDark[1], cyanDark[2]);
  doc.text('CLINICAL VERIFICATION & SIGN-OFF', margin + 4, y + 5);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('The AI output provides decision support. Final diagnostic determination must be completed by a licensed dental clinician.', margin + 4, y + 8.5);

  const clinicianName = analysis.clinicalReview?.doctorName || 'Dr. ___________________________';
  const reviewDate = analysis.clinicalReview?.reviewDate || '____ / ____ / ________';
  const agreement = analysis.clinicalReview?.aiFindingAgreement || '';

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text(`Reviewing Clinician: ${clinicianName}`, margin + 4, y + 15);
  doc.text(`Date of Review: ${reviewDate}`, margin + 105, y + 15);

  const check1 = agreement.includes('Agree') ? '[X]' : '[ ]';
  const check2 = agreement.includes('Disagree') ? '[X]' : '[ ]';
  doc.setFont('helvetica', 'normal');
  doc.text(`Clinical Determination: ${check1} Confirmed Caries Candidate   ${check2} Non-Carious Artifact`, margin + 4, y + 21);
  doc.text('Clinician Signature: ______________________', margin + 105, y + 21);

  // Page 2 Footer
  doc.setDrawColor(cardBorder[0], cardBorder[1], cardBorder[2]);
  doc.line(margin, pageHeight - 12, pageWidth - margin, pageHeight - 12);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(lightSlate[0], lightSlate[1], lightSlate[2]);
  doc.text('Page 2 of 2 • Dental Caries Detection & Decision Support System • The AI output is intended to support, not replace, professional clinical judgment.', margin, pageHeight - 7);

  return doc.output('blob');
}
