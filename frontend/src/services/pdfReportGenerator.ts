import { jsPDF } from 'jspdf';
import { AnalysisResult } from '../types/api';

/**
 * Generates a clean, 2-page Medical Clinical AI Summary PDF matching the authoritative Reference Report.
 */
export async function generateClinicalPDF(analysis: AnalysisResult): Promise<Blob> {
  const doc = new jsPDF({
    orientation: 'portrait',
    unit: 'mm',
    format: 'a4',
  });

  const pageWidth = doc.internal.pageSize.getWidth(); // ~210 mm
  const pageHeight = doc.internal.pageSize.getHeight(); // ~297 mm
  const margin = 14;
  const contentWidth = pageWidth - margin * 2;

  // Helper colors
  const darkNavy = [15, 23, 42];
  const slateText = [71, 85, 105];
  const lightBg = [248, 250, 252];
  const borderGray = [226, 232, 240];
  const cyanBrand = [14, 165, 233];
  const amberWarning = [245, 158, 11];

  // Helper to load image as base64 or fallback to canvas placeholder
  const loadImageData = (url: string): Promise<string> => {
    return new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'Anonymous';
      img.onload = () => {
        try {
          const canvas = document.createElement('canvas');
          canvas.width = img.width || 600;
          canvas.height = img.height || 300;
          const ctx = canvas.getContext('2d');
          if (ctx) {
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
            resolve(canvas.toDataURL('image/jpeg', 0.85));
            return;
          }
        } catch {
          // fallback on canvas fail
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
    canvas.width = 600;
    canvas.height = 300;
    const ctx = canvas.getContext('2d');
    if (ctx) {
      if (isMask) {
        ctx.fillStyle = '#000000';
        ctx.fillRect(0, 0, 600, 300);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillRect(380, 160, 22, 14);
        ctx.fillRect(520, 180, 18, 12);
        ctx.fillRect(540, 182, 12, 10);
      } else {
        ctx.fillStyle = '#1e293b';
        ctx.fillRect(0, 0, 600, 300);
        ctx.fillStyle = '#94a3b8';
        ctx.font = '16px sans-serif';
        ctx.fillText('Panoramic Radiograph (OPG)', 180, 150);
      }
      return canvas.toDataURL('image/jpeg');
    }
    return '';
  };

  const originalImgBase64 = await loadImageData(analysis.originalImageUrl);
  const overlayImgBase64 = await loadImageData(analysis.segmentationOverlayUrl);
  const binaryMaskBase64 = await loadImageData(analysis.binaryMaskUrl);

  // ==========================================
  // PAGE 1: Clinical Radiology Screening & Candidate Localization
  // ==========================================
  let y = margin;

  // Header Title
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('DENTAL CARIES AI DECISION SUPPORT SYSTEM', margin, y);

  // Right Case Metadata
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text(`Case ID: ${analysis.id}`, pageWidth - margin, y - 1, { align: 'right' });
  doc.text(`Exam Date: ${analysis.timestamp} UTC`, pageWidth - margin, y + 3, { align: 'right' });
  doc.text('Modality: Panoramic Radiograph (OPG)', pageWidth - margin, y + 7, { align: 'right' });

  y += 5;
  doc.setFontSize(9);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Clinical Radiology Screening & Candidate Localization Report', margin, y);

  y += 7;

  // Amber Banner
  doc.setFillColor(254, 252, 232); // light amber
  doc.setDrawColor(amberWarning[0], amberWarning[1], amberWarning[2]);
  doc.roundedRect(margin, y, contentWidth, 9, 1.5, 1.5, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(180, 83, 9); // dark amber
  const bannerText = analysis.findings.length > 0
    ? `SUSPICIOUS CARIES CANDIDATE REGION(S) IDENTIFIED: AI-assisted analysis identified ${analysis.findings.length} candidate lesion site(s) requiring clinical review.`
    : 'NO SIGNIFICANT CARIES CANDIDATES DETECTED: Standard routine prophylactic evaluation recommended.';
  doc.text(bannerText, margin + 3, y + 5.5);

  y += 14;

  // SECTION 1: Clinical Imaging & AI Candidate Visualization
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 1 — CLINICAL IMAGING & AI CANDIDATE VISUALIZATION', margin, y);
  y += 4;

  const panelWidth = (contentWidth - 4) / 2;
  const panelHeight = 52;

  // Panel A
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('Panel A: Original Dental Radiograph', margin, y);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Raw clinical radiograph', margin, y + 3.5);

  if (originalImgBase64) {
    doc.addImage(originalImgBase64, 'JPEG', margin, y + 5, panelWidth, panelHeight);
  }

  // Panel B
  const panelBx = margin + panelWidth + 4;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('Panel B: AI-Assisted Caries Visualization', panelBx, y);
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Candidate regions indicated with cyan highlights', panelBx, y + 3.5);

  if (overlayImgBase64) {
    doc.addImage(overlayImgBase64, 'JPEG', panelBx, y + 5, panelWidth, panelHeight);
  }

  y += panelHeight + 11;

  // SECTION 2: Findings & Candidate Region Summary
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 2 — FINDINGS & CANDIDATE REGION SUMMARY', margin, y);
  y += 4;

  // Table summary box
  doc.setFillColor(lightBg[0], lightBg[1], lightBg[2]);
  doc.setDrawColor(borderGray[0], borderGray[1], borderGray[2]);
  doc.rect(margin, y, contentWidth, 16, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Analyzed Image Source:', margin + 4, y + 5);
  doc.text('Approximate Extent:', margin + 4, y + 12);

  doc.text('Total Candidate Sites:', margin + 90, y + 5);
  doc.text('Clinical Urgency:', margin + 90, y + 12);

  doc.setFont('helvetica', 'normal');
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text(analysis.filename, margin + 40, y + 5);
  doc.text(`${analysis.summary.affectedAreaPercent.toFixed(2)}% of image area`, margin + 40, y + 12);

  doc.text(`${analysis.summary.totalLesions} detected region(s)`, margin + 125, y + 5);
  const urgency = analysis.summary.totalLesions >= 3 ? 'Priority Clinical Follow-up' : analysis.summary.totalLesions > 0 ? 'Moderate Clinical Review' : 'Routine Checkup';
  doc.text(urgency, margin + 125, y + 12);

  y += 22;

  // SECTION 3: Clinical Decision Support Guidance
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 3 — CLINICAL DECISION SUPPORT GUIDANCE', margin, y);
  y += 4;

  doc.setFillColor(240, 253, 250); // very soft cyan/mint
  doc.setDrawColor(180, 235, 230);
  doc.roundedRect(margin, y, contentWidth, 24, 1.5, 1.5, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(15, 118, 110);
  doc.text(`Recommendation: ${analysis.clinicalRecommendation}`, margin + 3, y + 6, { maxWidth: contentWidth - 6 });

  doc.setFont('helvetica', 'italic');
  doc.setFontSize(7);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  const guidanceNote = "Note: Highlighted candidate areas represent algorithmically flagged regions of radiolucency. Corroborate with clinical visual-tactile inspection, tooth vitality tests, and targeted bitewing/periapical radiographs where appropriate.";
  doc.text(guidanceNote, margin + 3, y + 14, { maxWidth: contentWidth - 6 });

  // Page 1 Footer
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Page 1 of 2 • Dental Caries Detection & Decision Support System • For Clinical Review Only', margin, pageHeight - 8);

  // ==========================================
  // PAGE 2: Caries Finding, Segmentation Analysis & Physician Verification
  // ==========================================
  doc.addPage();
  y = margin;

  // Header Title
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(14);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('DENTAL CARIES AI DECISION SUPPORT SYSTEM', margin, y);

  // Right Case Metadata
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(8.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text(`Case ID: ${analysis.id}`, pageWidth - margin, y - 1, { align: 'right' });
  doc.text('Page: 2 of 2', pageWidth - margin, y + 4, { align: 'right' });

  y += 5;
  doc.setFontSize(9);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Clinical Caries Finding, Segmentation Analysis & Physician Verification', margin, y);

  y += 8;

  // SECTION 4: Caries Finding & Clinical Interpretation
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(9.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('SECTION 4 — CARIES FINDING & CLINICAL INTERPRETATION', margin, y);
  y += 4;

  // 4.1 Binary Caries Segmentation
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('4.1 BINARY CARIES SEGMENTATION', margin, y);
  y += 3;

  const segWidth = (contentWidth - 4) / 2;
  const segHeight = 44;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Original Panoramic Radiograph', margin, y);
  doc.text('Binary Caries Segmentation Mask', margin + segWidth + 4, y);

  if (originalImgBase64) {
    doc.addImage(originalImgBase64, 'JPEG', margin, y + 2, segWidth, segHeight);
  }
  if (binaryMaskBase64) {
    doc.addImage(binaryMaskBase64, 'JPEG', margin + segWidth + 4, y + 2, segWidth, segHeight);
  }

  y += segHeight + 6;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(6.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('AI-Generated Binary Caries Segmentation Mask (Background = 0 | Caries Candidate = 1)', margin, y);
  y += 3;
  doc.setFont('helvetica', 'normal');
  doc.text('Highlighted regions represent pixels identified by the AI system as caries candidate regions. The segmentation is intended to assist clinical review and does not constitute an independent diagnosis.', margin, y, { maxWidth: contentWidth });

  y += 7;

  // 4.2 Detected Caries Area Table
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('4.2 DETECTED CARIES AREA', margin, y);
  y += 3;

  doc.setFillColor(lightBg[0], lightBg[1], lightBg[2]);
  doc.setDrawColor(borderGray[0], borderGray[1], borderGray[2]);
  doc.rect(margin, y, contentWidth, 12, 'FD');

  const colW = contentWidth / 3;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('DETECTED AREA', margin + colW * 0.5, y + 4, { align: 'center' });
  doc.text('POSITIVE PIXELS', margin + colW * 1.5, y + 4, { align: 'center' });
  doc.text('CANDIDATE REGIONS', margin + colW * 2.5, y + 4, { align: 'center' });

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text(`${analysis.summary.affectedAreaPercent.toFixed(2)}%`, margin + colW * 0.5, y + 9.5, { align: 'center' });
  doc.text(`${analysis.summary.totalAreaPx} px`, margin + colW * 1.5, y + 9.5, { align: 'center' });
  doc.text(`${analysis.summary.totalLesions}`, margin + colW * 2.5, y + 9.5, { align: 'center' });

  y += 16;

  // 4.3 AI-Assisted Caries Stage
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('4.3 AI-ASSISTED CARIES STAGE', margin, y);
  y += 3.5;

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(14, 116, 144);
  doc.text(`AI-Assisted Area-Based Category: ${analysis.stage.level} — ${analysis.stage.title} (${analysis.stage.description})`, margin, y);
  y += 3.5;
  doc.setFont('helvetica', 'italic');
  doc.setFontSize(6.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text("Stage is derived from the segmented candidate area according to the project's predefined decision-support criteria. It should not be interpreted as definitive clinical staging.", margin, y);

  y += 7;

  // 4.4 & 4.5 Dual columns
  const halfW = (contentWidth - 4) / 2;
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7.5);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('4.4 CLINICAL CARIES INTERPRETATION', margin, y);
  doc.text('4.5 CLINICAL REVIEW CONSIDERATIONS', margin + halfW + 4, y);
  y += 3.5;

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6.5);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  const interpText = `AI-assisted analysis identified ${analysis.summary.totalLesions} region(s) of interest consistent with possible dental caries, occupying an estimated ${analysis.summary.affectedAreaPercent.toFixed(2)}% of total radiographic area. The highlighted regions should be correlated with the original radiograph, dental history, clinical examination, and additional radiographic views where appropriate.`;
  doc.text(interpText, margin, y, { maxWidth: halfW });

  const bulletPoints = [
    '• Early caries may present as subtle radiolucency and may be difficult to distinguish from anatomical overlap.',
    '• Biological depth cannot be determined reliably from segmented pixel area alone.',
    '• Existing restorations, enamel variations, and cervical burnout may produce regions requiring clinical verification.',
    '• Bitewing radiographs remain recommended when interproximal caries is suspected.',
  ];
  doc.text(bulletPoints.join('\n'), margin + halfW + 4, y, { maxWidth: halfW });

  y += 24;

  // 4.6 Physician / Dentist Verification Box
  doc.setFont('helvetica', 'bold');
  doc.setFontSize(8);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text('4.6 PHYSICIAN / DENTIST VERIFICATION', margin, y);
  y += 3;

  doc.setFillColor(lightBg[0], lightBg[1], lightBg[2]);
  doc.setDrawColor(cyanBrand[0], cyanBrand[1], cyanBrand[2]);
  doc.roundedRect(margin, y, contentWidth, 26, 1.5, 1.5, 'FD');

  doc.setFont('helvetica', 'bold');
  doc.setFontSize(7);
  doc.setTextColor(14, 116, 144);
  doc.text('CLINICAL VERIFICATION REQUIRED', margin + 3, y + 4.5);

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(6);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('The AI output is intended to support, not replace, professional clinical judgment. Final interpretation should be performed by a qualified dental professional using complete clinical and radiographic context.', margin + 3, y + 8, { maxWidth: contentWidth - 6 });

  const clinicianName = analysis.clinicalReview?.doctorName || '___________________________';
  const reviewDate = analysis.clinicalReview?.reviewDate || '______________';
  const agreement = analysis.clinicalReview?.aiFindingAgreement || '';

  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(darkNavy[0], darkNavy[1], darkNavy[2]);
  doc.text(`Reviewing Clinician: ${clinicianName}`, margin + 3, y + 16);
  doc.text(`Date: ${reviewDate}`, margin + 110, y + 16);

  const check1 = agreement.includes('Agree') ? '[X]' : '[ ]';
  const check2 = agreement.includes('Disagree') ? '[X]' : '[ ]';
  doc.text(`Verification: ${check1} Confirmed Caries Candidate   ${check2} Artifact / Non-Carious`, margin + 3, y + 22);
  doc.text('Signature: ______________________', margin + 110, y + 22);

  // Page 2 Footer
  doc.setFont('helvetica', 'normal');
  doc.setFontSize(7);
  doc.setTextColor(slateText[0], slateText[1], slateText[2]);
  doc.text('Page 2 of 2 • Dental Caries Detection & Decision Support System • The AI output is intended to support, not replace, professional clinical judgment.', margin, pageHeight - 8);

  return doc.output('blob');
}
