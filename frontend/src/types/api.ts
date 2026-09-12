export type CariesDepth = 'Enamel (E1/E2)' | 'Dentin (D1/D2)' | 'Deep Dentin / Pulp (D3)' | 'Incipient' | 'Enamel (E2) / Outer Dentin';

export interface LesionFinding {
  id: string;
  lesionNumber: string; // e.g. "Lesion 01", "L1", "L2"
  toothNumberFDI?: string; // e.g. "16", "24", "36", "46", "47"
  location: 'Upper Right' | 'Upper Left' | 'Lower Left' | 'Lower Right' | 'Anterior' | 'Posterior' | 'Middle-Left' | 'Upper-Right' | 'Lower-Right' | string;
  depth: CariesDepth | string;
  confidence: number; // 0.0 - 1.0 (e.g. 0.94 -> 94%)
  pixelArea: number; // e.g. 112
  areaPercent: number; // e.g. 0.04%
  stage: 'Stage 0' | 'Stage 1' | 'Stage 2' | 'Stage 3' | 'Level 1' | 'Level 2' | string;
  bbox: [number, number, number, number]; // [x, y, w, h] normalized or relative to image
  clinicalRecommendation: string;
}

export interface ClinicalReviewData {
  doctorName: string;
  reviewDate: string;
  clinicalAssessment: 
    | 'No caries identified'
    | 'Early caries suspected'
    | 'Moderate caries suspected'
    | 'Extensive caries suspected'
    | 'Requires further examination';
  aiFindingAgreement: 
    | 'Agree with AI finding'
    | 'Partially agree'
    | 'Disagree'
    | 'Unable to determine';
  recommendedFollowUp: 
    | 'Routine dental examination'
    | 'Clinical examination recommended'
    | 'Further radiographic evaluation recommended'
    | 'Immediate clinical review recommended';
  additionalNotes?: string;
  isVerified?: boolean;
  verifiedAt?: string;
}

export interface EvaluationMetrics {
  diceScore?: number;
  iou?: number;
  precision?: number;
  recall?: number;
  f1Score?: number;
  specificity?: number;
  accuracy?: number;
  benchmarkReference?: string;
}

export interface AnalysisResult {
  id: string;
  timestamp: string;
  filename: string;
  patientPseudoId?: string;
  originalImageUrl: string;
  segmentationOverlayUrl: string;
  binaryMaskUrl: string; // Caries = WHITE, Background = BLACK
  overallFinding: string;
  stage: {
    level: string;
    title: string;
    description: string;
    color: 'emerald' | 'cyan' | 'amber' | 'rose' | string;
  };
  summary: {
    totalAreaPx: number;
    affectedAreaPercent: number;
    totalLesions: number;
    meanConfidence: number;
    highestConfidence: number;
    regionsRequiringReview: number;
  };
  findings: LesionFinding[];
  evaluationMetrics?: EvaluationMetrics;
  clinicalReview?: ClinicalReviewData;
  clinicalRecommendation: string;
  inferenceTimeMs: number;
}

export interface HealthStatus {
  status: 'ONLINE' | 'OFFLINE' | 'DEGRADED';
  isLiveBackend: boolean;
  serviceName: string;
  version: string;
  device?: string;
}

export interface HistoryItem {
  id: string;
  timestamp: string;
  filename: string;
  patientId: string;
  overallFinding: string;
  stageLevel: string;
  lesionCount: number;
  affectedAreaPercent: number;
  meanConfidence: number;
  isReviewed: boolean;
  status: 'COMPLETED' | 'PENDING' | 'FAILED';
}

export interface ReportItem {
  id: string;
  analysisId: string;
  timestamp: string;
  patientPseudoId: string;
  filename: string;
  overallFinding: string;
  stageLevel: string;
  lesionCount: number;
  affectedAreaPercent: number;
  reviewStatus: 'Pending Review' | 'Clinically Verified' | 'Reviewed with Edits' | string;
}

export interface AssistantMessage {
  id: string;
  sender: 'user' | 'assistant';
  text: string;
  timestamp: string;
  detectedEmotion?: 'anxious' | 'worried' | 'confused' | 'scared' | 'frustrated' | 'neutral' | 'reassured';
}
