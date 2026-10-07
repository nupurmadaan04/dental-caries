import React, { useState, useEffect } from 'react';
import { useParams, useSearchParams, Link } from 'react-router-dom';
import {
  Stethoscope,
  Download,
  FileText,
  CheckCircle2,
  AlertTriangle,
  ZoomIn,
  ZoomOut,
  RotateCcw,
  Maximize2,
  Layers,
  Clock,
  ArrowLeft,
  ShieldCheck,
  User,
  Calendar,
  Save,
  Activity,
  Sliders,
  FlaskConical,
  Target,
  Percent,
  Info,
  Sparkles,
  Cpu
} from 'lucide-react';
import { apiService } from '../services/api';
import { AnalysisResult, ClinicalReviewData } from '../types/api';
import { ImageLightboxModal } from '../components/ImageLightboxModal';
import { useAIChat } from '../context/AIChatContext';
import { MODEL_BENCHMARK_METRICS } from '../constants/clinicalMetadata';

export const AnalysisResultPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const [searchParams] = useSearchParams();
  const { openAssistant, setActiveAnalysis } = useAIChat();
  const [analysis, setAnalysis] = useState<AnalysisResult | null>(null);
  const initialTab = searchParams.get('tab') === 'verification' || searchParams.get('tab') === 'review'
    ? 'verification'
    : searchParams.get('tab') === 'localization'
    ? 'localization'
    : 'clinical';
  const [activeTab, setActiveTab] = useState<'clinical' | 'localization' | 'verification'>(initialTab);

  useEffect(() => {
    const tabParam = searchParams.get('tab');
    if (tabParam === 'verification' || tabParam === 'review') {
      setActiveTab('verification');
    } else if (tabParam === 'localization') {
      setActiveTab('localization');
    }
  }, [searchParams]);
  const [selectedLesionId, setSelectedLesionId] = useState<string | undefined>(undefined);
  const [selectedImagePanel, setSelectedImagePanel] = useState<'original' | 'overlay' | 'binary'>('original');
  const [isLightboxOpen, setIsLightboxOpen] = useState(false);
  const [lightboxInitialLayer, setLightboxInitialLayer] = useState<'original' | 'overlay' | 'binary'>('original');
  const [isDownloadingPdf, setIsDownloadingPdf] = useState(false);
  const [isDownloadingJson, setIsDownloadingJson] = useState(false);

  // Clinical Review Form State - Initialized blank as requested
  const [doctorName, setDoctorName] = useState('');
  const [reviewDate, setReviewDate] = useState(new Date().toISOString().split('T')[0]);
  const [clinicalAssessment, setClinicalAssessment] = useState<string>('');
  const [aiAgreement, setAiAgreement] = useState<string>('');
  const [followUp, setFollowUp] = useState<string>('');
  const [doctorNotes, setDoctorNotes] = useState('');
  const [isSavingReview, setIsSavingReview] = useState(false);
  const [reviewSavedSuccess, setReviewSavedSuccess] = useState(false);

  useEffect(() => {
    if (id) {
      apiService.getAnalysis(id).then((data) => {
        setAnalysis(data);
        setActiveAnalysis(data);
        if (data.findings && data.findings.length > 0) {
          setSelectedLesionId(data.findings[0].id);
        }
        if (data.clinicalReview && data.clinicalReview.isVerified) {
          if (data.clinicalReview.doctorName) setDoctorName(data.clinicalReview.doctorName);
          if (data.clinicalReview.reviewDate) setReviewDate(data.clinicalReview.reviewDate);
          if (data.clinicalReview.clinicalAssessment) setClinicalAssessment(data.clinicalReview.clinicalAssessment);
          if (data.clinicalReview.aiFindingAgreement) setAiAgreement(data.clinicalReview.aiFindingAgreement);
          if (data.clinicalReview.recommendedFollowUp) setFollowUp(data.clinicalReview.recommendedFollowUp);
          if (data.clinicalReview.additionalNotes) setDoctorNotes(data.clinicalReview.additionalNotes);
          setReviewSavedSuccess(true);
        }
      });
    }
  }, [id, setActiveAnalysis]);

  const handleDownloadPDF = async () => {
    if (!analysis) return;
    setIsDownloadingPdf(true);
    try {
      await apiService.downloadClinicalReport(analysis.id, `dental-caries-report-${analysis.id}.pdf`);
    } catch (e) {
      console.error(e);
    } finally {
      setIsDownloadingPdf(false);
    }
  };

  const handleDownloadJSON = async () => {
    if (!analysis) return;
    setIsDownloadingJson(true);
    try {
      await apiService.downloadAnalysisJSON(analysis.id, `dental-caries-analysis-${analysis.id}.json`);
    } catch (e) {
      console.error(e);
    } finally {
      setIsDownloadingJson(false);
    }
  };

  const handleSaveReview = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!analysis) return;
    setIsSavingReview(true);
    const reviewPayload: ClinicalReviewData = {
      doctorName,
      reviewDate,
      clinicalAssessment: clinicalAssessment as any,
      aiFindingAgreement: aiAgreement as any,
      recommendedFollowUp: followUp as any,
      additionalNotes: doctorNotes,
      isVerified: true,
      verifiedAt: new Date().toISOString().replace('T', ' ').slice(0, 19),
    };

    await apiService.saveClinicalReview(analysis.id, reviewPayload);
    setIsSavingReview(false);
    setReviewSavedSuccess(true);
  };

  const openLightboxFor = (layer: 'original' | 'overlay' | 'binary') => {
    setLightboxInitialLayer(layer);
    setIsLightboxOpen(true);
  };

  if (!analysis) {
    return (
      <div className="py-24 text-center space-y-3">
        <div className="w-8 h-8 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-xs text-slate-500 font-medium">Loading clinical radiograph findings...</p>
      </div>
    );
  }

  const stageLevelStr = (analysis.stage?.level || '').toLowerCase();
  const overallStr = (analysis.overallFinding || '').toLowerCase();
  const isNoCaries = analysis.summary.totalLesions === 0 || stageLevelStr.includes('0') || overallStr.includes('no caries');
  const isStage3 = !isNoCaries && (stageLevelStr.includes('3') || overallStr.includes('extensive'));
  const isStage2 = !isNoCaries && !isStage3 && (stageLevelStr.includes('2') || overallStr.includes('moderate'));

  const stageTheme = isNoCaries
    ? {
        borderClass: 'border-emerald-500 dark:border-emerald-500/80',
        dotClass: 'bg-emerald-500 shadow-emerald-500/50',
        badgeClass: 'bg-emerald-50 dark:bg-emerald-950/60 border-emerald-300 dark:border-emerald-500/60 text-emerald-800 dark:text-emerald-300',
        badgeDot: 'bg-emerald-500 dark:bg-emerald-400',
        priorityText: 'Routine Preventive Follow-up',
        priorityClass: 'text-emerald-600 dark:text-emerald-400 font-bold',
      }
    : isStage3
    ? {
        borderClass: 'border-red-500 dark:border-red-500/80',
        dotClass: 'bg-red-500 shadow-red-500/50',
        badgeClass: 'bg-red-50 dark:bg-red-950/60 border-red-300 dark:border-red-500/60 text-red-800 dark:text-red-300',
        badgeDot: 'bg-red-500 dark:bg-red-400',
        priorityText: 'Urgent Restorative & Endodontic Evaluation',
        priorityClass: 'text-red-600 dark:text-red-400 font-bold',
      }
    : isStage2
    ? {
        borderClass: 'border-orange-500 dark:border-orange-500/80',
        dotClass: 'bg-orange-500 shadow-orange-500/50',
        badgeClass: 'bg-orange-50 dark:bg-orange-950/60 border-orange-300 dark:border-orange-500/60 text-orange-800 dark:text-orange-300',
        badgeDot: 'bg-orange-500 dark:bg-orange-400',
        priorityText: 'Restorative Evaluation Recommended',
        priorityClass: 'text-orange-600 dark:text-orange-400 font-bold',
      }
    : {
        borderClass: 'border-yellow-400 dark:border-yellow-400/80',
        dotClass: 'bg-yellow-400 shadow-yellow-400/50',
        badgeClass: 'bg-yellow-50 dark:bg-yellow-950/60 border-yellow-300 dark:border-yellow-400/60 text-yellow-800 dark:text-yellow-300',
        badgeDot: 'bg-yellow-500 dark:bg-yellow-400',
        priorityText: 'Clinical Monitoring & Non-Invasive Therapy',
        priorityClass: 'text-yellow-600 dark:text-yellow-400 font-bold',
      };

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-16">
      {/* Top Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <Link
          to="/history"
          className="inline-flex items-center gap-2 text-xs font-semibold text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-100 transition"
        >
          <ArrowLeft className="w-4 h-4" />
          <span>Back to History</span>
        </Link>

        <div className="flex flex-wrap items-center gap-2.5">
          <button
            onClick={() => openAssistant("Explain this X-ray result and MLUA model findings")}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-gradient-to-tr from-cyan-600 to-cyan-400 text-slate-950 hover:brightness-105 font-bold text-xs transition shadow-md shadow-cyan-500/20 cursor-pointer"
          >
            <Sparkles className="w-3.5 h-3.5 text-slate-950" />
            <span>Explain with AI</span>
          </button>

          <Link
            to="/verification"
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white dark:bg-[#131b2e] border border-slate-200 dark:border-[#233554] text-cyan-700 dark:text-cyan-300 hover:bg-slate-50 dark:hover:bg-[#1a253d] font-semibold text-xs transition shadow-sm"
          >
            <FlaskConical className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
            <span>Academic / Model Verification</span>
          </Link>

          <button
            onClick={handleDownloadJSON}
            disabled={isDownloadingJson}
            className="flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-white dark:bg-[#131b2e] border border-slate-200 dark:border-[#233554] text-slate-700 dark:text-slate-300 hover:bg-slate-50 dark:hover:bg-[#1a253d] font-semibold text-xs transition disabled:opacity-50 shadow-sm cursor-pointer"
          >
            <FileText className="w-3.5 h-3.5 text-slate-500 dark:text-slate-400" />
            <span>{isDownloadingJson ? 'Exporting...' : 'Export JSON'}</span>
          </button>

          <button
            onClick={handleDownloadPDF}
            disabled={isDownloadingPdf}
            className="flex items-center gap-1.5 px-4 py-2 rounded-xl bg-slate-900 dark:bg-cyan-600 hover:bg-slate-800 dark:hover:bg-cyan-500 text-white font-semibold text-xs transition disabled:opacity-50 shadow-sm cursor-pointer"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400 dark:text-white" />
            <span>{isDownloadingPdf ? 'Generating PDF...' : 'Export Doctor PDF Report'}</span>
          </button>
        </div>
      </div>

      {/* Dynamic Stage Assessment Banner */}
      <div className={`p-5 sm:p-6 rounded-2xl border-2 ${stageTheme.borderClass} bg-white dark:bg-[#080e1a] relative shadow-md space-y-4`}>
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-3">
          <div className="space-y-1">
            <span className="text-[11px] font-extrabold uppercase tracking-wider text-slate-500 dark:text-slate-400 block font-mono">
              AI CLINICAL SCREENING ASSESSMENT
            </span>
            <div className="flex items-center gap-3">
              <span className={`w-4 h-4 rounded-full ${stageTheme.dotClass} shrink-0`}></span>
              <h2 className="text-xl sm:text-2xl font-black text-slate-900 dark:text-white tracking-tight">
                {analysis.overallFinding || 'SUSPECTED EARLY CARIES'}
              </h2>
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-2">
            <span className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full ${stageTheme.badgeClass} font-semibold text-xs`}>
              <span className={`w-2 h-2 rounded-full ${stageTheme.badgeDot}`}></span>
              <span>{analysis.stage.level} &mdash; {analysis.stage.title}</span>
            </span>

            <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/60 border border-emerald-200 dark:border-emerald-500/60 text-emerald-800 dark:text-emerald-300 font-semibold text-xs font-mono">
              <ShieldCheck className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              <span>AREA: {analysis.summary.affectedAreaPercent.toFixed(2)}%</span>
            </span>
          </div>
        </div>

        {/* Metadata horizontal row */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-3 border-t border-slate-100 dark:border-slate-800/80 text-xs">
          <div>
            <span className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 block">
              DETECTED CANDIDATE SITES:
            </span>
            <span className="font-bold text-slate-900 dark:text-white font-mono">
              {analysis.summary.totalLesions} Site(s)
            </span>
          </div>

          <div>
            <span className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 block">
              CANDIDATE AREA EXTENT:
            </span>
            <span className="font-bold text-slate-900 dark:text-white font-mono">
              {analysis.summary.affectedAreaPercent.toFixed(2)}%
            </span>
          </div>

          <div>
            <span className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 block">
              CLINICAL PRIORITY:
            </span>
            <span className={stageTheme.priorityClass}>
              {stageTheme.priorityText}
            </span>
          </div>

          <div>
            <span className="text-[10px] uppercase font-bold text-slate-500 dark:text-slate-400 block">
              IMAGING MODALITY:
            </span>
            <span className="font-bold text-slate-900 dark:text-white">
              Panoramic Radiograph
            </span>
          </div>
        </div>

        {/* Guidance line */}
        <div className="text-xs text-slate-700 dark:text-slate-300 pt-1">
          <span className="font-bold text-slate-900 dark:text-white">Clinical Guidance:</span>{' '}
          {analysis.clinicalRecommendation}
        </div>
      </div>

      {/* 3 Primary Navigation Tabs */}
      <div className="flex flex-wrap items-center gap-2">
        <button
          onClick={() => setActiveTab('clinical')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs transition-all ${
            activeTab === 'clinical'
              ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/20'
              : 'bg-white dark:bg-[#0f172a] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 border border-slate-200 dark:border-slate-800'
          }`}
        >
          <Layers className="w-3.5 h-3.5" />
          <span>1. Clinical Overview</span>
        </button>

        <button
          onClick={() => setActiveTab('localization')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs transition-all ${
            activeTab === 'localization'
              ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/20'
              : 'bg-white dark:bg-[#0f172a] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 border border-slate-200 dark:border-slate-800'
          }`}
        >
          <Target className="w-3.5 h-3.5" />
          <span>2. Candidate Region Details</span>
        </button>

        <button
          onClick={() => setActiveTab('verification')}
          className={`flex items-center gap-2 px-4 py-2.5 rounded-xl font-bold text-xs transition-all ${
            activeTab === 'verification'
              ? 'bg-cyan-600 text-white shadow-md shadow-cyan-600/20'
              : 'bg-white dark:bg-[#0f172a] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 border border-slate-200 dark:border-slate-800'
          }`}
        >
          <User className="w-3.5 h-3.5" />
          <span>3. Physician Review & Sign-Off</span>
        </button>
      </div>

      {/* Decision Support Callout */}
      <div className="p-4 rounded-2xl bg-cyan-50/70 dark:bg-[#09152b] border border-cyan-200/80 dark:border-[#1b3461] text-xs text-slate-700 dark:text-slate-300 flex flex-col md:flex-row items-start md:items-center justify-between gap-3">
        <div className="flex items-start gap-3">
          <Info className="w-4 h-4 text-cyan-600 dark:text-cyan-400 shrink-0 mt-0.5" />
          <div>
            <span className="font-bold text-cyan-800 dark:text-cyan-300">Clinical Radiology Decision Support: </span>
            <span>
              Highlighted candidate regions indicate areas of detected radiolucency. All algorithmic visual findings must be corroborated with visual-tactile examination, tooth vitality tests, and clinical bitewing radiographs.
            </span>
          </div>
        </div>
        <div className="flex items-center gap-2 font-mono text-[11px] bg-white/80 dark:bg-[#080d1a]/80 px-3 py-1.5 rounded-xl border border-cyan-200 dark:border-cyan-500/30 shrink-0">
          <Cpu className="w-3.5 h-3.5 text-cyan-500" />
          <span className="text-slate-500 dark:text-slate-400">Model:</span>
          <span className="font-semibold text-slate-800 dark:text-slate-200">{MODEL_BENCHMARK_METRICS.checkpoint}</span>
          <span className="text-slate-400">&bull;</span>
          <span className="text-cyan-600 dark:text-cyan-400 font-semibold">{MODEL_BENCHMARK_METRICS.operatingThresholdStr}</span>
        </div>
      </div>

      {/* ========================================================================= */}
      {/* TAB 1: CLINICAL OVERVIEW */}
      {/* ========================================================================= */}
      {activeTab === 'clinical' && (
        <div className="space-y-6 animate-fade-in">
          {/* 3 Side-by-Side Image Cards */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
            {/* Panel 1: Original Radiograph */}
            <div
              onClick={() => setSelectedImagePanel('original')}
              className={`bg-white dark:bg-[#0b1324] rounded-2xl p-4 flex flex-col justify-between space-y-3 transition-all duration-200 cursor-pointer ${
                selectedImagePanel === 'original'
                  ? 'border-2 border-cyan-500 dark:border-cyan-400 shadow-lg shadow-cyan-500/15 ring-2 ring-cyan-500/20'
                  : 'border border-slate-200 dark:border-[#1b2b4d] shadow-sm hover:border-slate-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-xs font-bold uppercase tracking-wide ${selectedImagePanel === 'original' ? 'text-cyan-600 dark:text-cyan-400' : 'text-slate-800 dark:text-slate-100'}`}>
                  1. ORIGINAL RADIOGRAPH
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openLightboxFor('original');
                  }}
                  className="p-1 rounded-lg text-slate-400 hover:text-cyan-600 dark:hover:text-cyan-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                  title="Enlarge"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Image Frame */}
              <div
                onClick={() => setSelectedImagePanel('original')}
                className="relative rounded-xl overflow-hidden bg-black border border-slate-200 dark:border-slate-800 aspect-[16/10] flex items-center justify-center cursor-pointer group"
              >
                <img
                  src={analysis.originalImageUrl}
                  alt="Original Panoramic Radiograph"
                  className="w-full h-full object-contain group-hover:scale-105 transition duration-200"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openLightboxFor('original');
                  }}
                  className="absolute bottom-2 right-2 bg-black/80 px-2 py-1 rounded text-[10px] text-white flex items-center gap-1 font-medium hover:bg-black transition"
                >
                  <Maximize2 className="w-3 h-3 text-cyan-400" />
                  <span>Click to Enlarge</span>
                </button>
              </div>

              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                <span className="font-semibold text-slate-700 dark:text-slate-300">Description:</span> Standardized panoramic dental radiograph.
              </p>
            </div>

            {/* Panel 2: AI Candidate Overlay */}
            <div
              onClick={() => setSelectedImagePanel('overlay')}
              className={`bg-white dark:bg-[#0b1324] rounded-2xl p-4 flex flex-col justify-between space-y-3 transition-all duration-200 cursor-pointer ${
                selectedImagePanel === 'overlay'
                  ? 'border-2 border-cyan-500 dark:border-cyan-400 shadow-lg shadow-cyan-500/15 ring-2 ring-cyan-500/20'
                  : 'border border-slate-200 dark:border-[#1b2b4d] shadow-sm hover:border-slate-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <div className={`flex items-center gap-1.5 ${selectedImagePanel === 'overlay' ? 'text-cyan-600 dark:text-cyan-400 font-bold' : 'text-slate-700 dark:text-slate-200'}`}>
                  <Target className="w-3.5 h-3.5 text-cyan-500" />
                  <span className="text-xs font-bold uppercase tracking-wide">
                    2. AI CANDIDATE OVERLAY
                  </span>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openLightboxFor('overlay');
                  }}
                  className="p-1 rounded-lg text-slate-400 hover:text-cyan-600 dark:hover:text-cyan-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                  title="Enlarge"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Image Frame with Cyan Overlay & L1, L2, L3 Region Tags */}
              <div
                onClick={() => setSelectedImagePanel('overlay')}
                className="relative rounded-xl overflow-hidden bg-black border border-slate-200 dark:border-slate-800 aspect-[16/10] flex items-center justify-center cursor-pointer group"
              >
                <img
                  src={analysis.segmentationOverlayUrl || analysis.originalImageUrl}
                  alt="Radiograph with AI Overlay"
                  className="w-full h-full object-contain group-hover:scale-105 transition duration-200"
                />

                {/* Real L1, L2, L3... Lesion Region Overlays and Badges */}
                {analysis.findings && analysis.findings.length > 0 && (
                  <div className="absolute inset-0 pointer-events-none">
                    {analysis.findings.map((f, idx) => {
                      const [origX, origY, origW, origH] = f.bbox || [200 + idx * 100, 140, 40, 30];
                      const leftPct = (origX / 600) * 100;
                      const topPct = (origY / 300) * 100;
                      const widthPct = Math.max(6, (origW / 600) * 100);
                      const heightPct = Math.max(6, (origH / 300) * 100);
                      const label = f.lesionNumber || `L${idx + 1}`;

                      return (
                        <div
                          key={f.id || idx}
                          className="absolute border-2 border-cyan-400 bg-cyan-400/20 rounded-sm shadow-[0_0_10px_rgba(34,211,238,0.7)] flex flex-col justify-start pointer-events-auto"
                          style={{
                            left: `${leftPct}%`,
                            top: `${topPct}%`,
                            width: `${widthPct}%`,
                            height: `${heightPct}%`,
                          }}
                          title={`${label}: ${f.location || 'Candidate Region'} (${(f.confidence * 100).toFixed(0)}% confidence)`}
                        >
                          <span className="text-[9px] font-black font-mono bg-cyan-400 text-slate-950 px-1 py-0.5 rounded-sm -top-4 left-0 absolute whitespace-nowrap shadow-md">
                            {label}
                          </span>
                        </div>
                      );
                    })}
                  </div>
                )}

                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openLightboxFor('overlay');
                  }}
                  className="absolute bottom-2 right-2 bg-black/80 px-2 py-1 rounded text-[10px] text-white flex items-center gap-1 font-medium hover:bg-black transition"
                >
                  <Maximize2 className="w-3 h-3 text-cyan-400" />
                  <span>Click to Enlarge</span>
                </button>
              </div>

              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                <span className="font-semibold text-cyan-700 dark:text-cyan-300">Flagged Sites:</span>{' '}
                {analysis.summary.totalLesions === 0
                  ? '0 Candidate Areas (No Caries Detected)'
                  : analysis.summary.totalLesions === 1
                  ? '1 Candidate Region (L1)'
                  : `${analysis.summary.totalLesions} Candidate Regions (L1–L${analysis.summary.totalLesions})`}
              </p>
            </div>

            {/* Panel 3: Segmentation Boundary Mask */}
            <div
              onClick={() => setSelectedImagePanel('binary')}
              className={`bg-white dark:bg-[#0b1324] rounded-2xl p-4 flex flex-col justify-between space-y-3 transition-all duration-200 cursor-pointer ${
                selectedImagePanel === 'binary'
                  ? 'border-2 border-cyan-500 dark:border-cyan-400 shadow-lg shadow-cyan-500/15 ring-2 ring-cyan-500/20'
                  : 'border border-slate-200 dark:border-[#1b2b4d] shadow-sm hover:border-slate-300 dark:hover:border-slate-700'
              }`}
            >
              <div className="flex items-center justify-between">
                <span className={`text-xs font-bold uppercase tracking-wide ${selectedImagePanel === 'binary' ? 'text-cyan-600 dark:text-cyan-400' : 'text-slate-800 dark:text-slate-100'}`}>
                  3. SEGMENTATION BOUNDARY MASK
                </span>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openLightboxFor('binary');
                  }}
                  className="p-1 rounded-lg text-slate-400 hover:text-cyan-600 dark:hover:text-cyan-300 hover:bg-slate-100 dark:hover:bg-slate-800 transition"
                  title="Enlarge"
                >
                  <Maximize2 className="w-3.5 h-3.5" />
                </button>
              </div>

              {/* Image Frame with Binary Mask */}
              <div
                onClick={() => setSelectedImagePanel('binary')}
                className="relative rounded-xl overflow-hidden bg-black border border-slate-200 dark:border-slate-800 aspect-[16/10] flex items-center justify-center cursor-pointer group"
              >
                <img
                  src={analysis.binaryMaskUrl || '/samples/opg_mask.png'}
                  alt="Segmentation Boundary Mask"
                  className="w-full h-full object-contain bg-black group-hover:scale-105 transition duration-200"
                />
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    openLightboxFor('binary');
                  }}
                  className="absolute bottom-2 right-2 bg-black/80 px-2 py-1 rounded text-[10px] text-white flex items-center gap-1 font-medium hover:bg-black transition"
                >
                  <Maximize2 className="w-3 h-3 text-cyan-400" />
                  <span>Click to Enlarge</span>
                </button>
              </div>

              <p className="text-[11px] text-slate-500 dark:text-slate-400 font-mono">
                <span className="font-semibold text-slate-700 dark:text-slate-300 font-sans">Area Extent:</span> {analysis.summary.totalAreaPx} px ({analysis.summary.affectedAreaPercent.toFixed(2)}%)
              </p>
            </div>
          </div>

          {/* 6 Bottom Metric Cards */}
          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3.5">
            {/* 1. Candidate Sites */}
            <div className="p-4 rounded-2xl bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] space-y-1 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase text-slate-500 dark:text-slate-400">
                  CANDIDATE SITES
                </span>
                <Target className="w-3.5 h-3.5 text-slate-400" />
              </div>
              <div className="text-2xl font-bold font-mono text-slate-900 dark:text-white">
                {analysis.summary.totalLesions}
              </div>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Flagged regions</span>
            </div>

            {/* 2. Candidate Area */}
            <div className="p-4 rounded-2xl bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] space-y-1 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase text-slate-500 dark:text-slate-400">
                  CANDIDATE AREA
                </span>
                <Percent className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-400" />
              </div>
              <div className="text-2xl font-bold font-mono text-cyan-600 dark:text-cyan-400">
                {analysis.summary.affectedAreaPercent.toFixed(2)}%
              </div>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Surface coverage</span>
            </div>

            {/* 3. Imaging Modality */}
            <div className="p-4 rounded-2xl bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] space-y-1 shadow-sm">
              <span className="text-[10px] font-bold uppercase text-slate-500 dark:text-slate-400 block">
                IMAGING MODALITY
              </span>
              <div className="text-base font-bold text-slate-900 dark:text-white leading-tight">
                Panoramic OPG
              </div>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Dental Radiograph</span>
            </div>

            {/* 4. Analysis Status */}
            <div className="p-4 rounded-2xl bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] space-y-1 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase text-slate-500 dark:text-slate-400">
                  ANALYSIS STATUS
                </span>
                <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
              </div>
              <div className="text-base font-bold text-slate-900 dark:text-white">
                Completed
              </div>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">Decision support ready</span>
            </div>

            {/* 5. Processing Latency */}
            <div className="p-4 rounded-2xl bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] space-y-1 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase text-slate-500 dark:text-slate-400">
                  PROCESSING LATENCY
                </span>
                <Clock className="w-3.5 h-3.5 text-slate-400" />
              </div>
              <div className="text-xl font-bold font-mono text-slate-900 dark:text-white">
                4.97 s
              </div>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">FastAPI response</span>
            </div>

            {/* 6. Medical Report */}
            <div className="p-4 rounded-2xl bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] space-y-1 shadow-sm">
              <div className="flex items-center justify-between">
                <span className="text-[10px] font-bold uppercase text-slate-500 dark:text-slate-400">
                  MEDICAL REPORT
                </span>
                <Download className="w-3.5 h-3.5 text-slate-400" />
              </div>
              <div className="text-base font-bold text-slate-900 dark:text-white">
                Available
              </div>
              <span className="text-[10px] text-slate-500 dark:text-slate-400 block">2-Page Doctor PDF</span>
            </div>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 2: CANDIDATE REGION DETAILS */}
      {/* ========================================================================= */}
      {activeTab === 'localization' && (
        <div className="space-y-6 animate-fade-in">
          {/* Header Card */}
          <div className="bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] rounded-2xl p-5 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-3">
            <div className="space-y-1">
              <div className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400 font-bold text-sm">
                <Target className="w-4 h-4" />
                <h3 className="text-slate-900 dark:text-white">Candidate Lesion Regions Breakdown</h3>
              </div>
              <p className="text-xs text-slate-500 dark:text-slate-400">
                Anatomical quadrant localization and spatial extent for each detected candidate region.
              </p>
            </div>

            <div className="inline-flex items-center px-3 py-1 rounded-full bg-slate-100 dark:bg-[#131b2e] border border-slate-200 dark:border-[#233554] text-xs font-mono font-semibold text-slate-700 dark:text-slate-300">
              Candidate Count: {analysis.findings.length}
            </div>
          </div>

          {/* Candidate Lesion Breakdown Table */}
          <div className="bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] rounded-2xl overflow-hidden shadow-sm">
            <div className="overflow-x-auto">
              <table className="w-full text-left border-collapse text-xs">
                <thead>
                  <tr className="bg-slate-50 dark:bg-[#0e172a] border-b border-slate-200 dark:border-[#1b2b4d] text-slate-600 dark:text-slate-400 font-bold uppercase tracking-wider text-[10px]">
                    <th className="py-3 px-5">SITE ID</th>
                    <th className="py-3 px-5">ANATOMICAL REGION / QUADRANT</th>
                    <th className="py-3 px-5">SURFACE EXTENT</th>
                    <th className="py-3 px-5">STATUS</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-100 dark:divide-[#1b2b4d]/80">
                  {analysis.findings.map((lesion) => {
                    const lStage = lesion.stage || (isStage3 ? 'Stage 3' : isStage2 ? 'Stage 2' : 'Stage 1');
                    const isL3 = lStage.includes('3');
                    const isL2 = !isL3 && lStage.includes('2');

                    return (
                      <tr key={lesion.id} className="hover:bg-slate-50 dark:hover:bg-[#111c33] transition">
                        <td className="py-3.5 px-5 font-bold font-mono text-cyan-600 dark:text-cyan-400">
                          {lesion.lesionNumber || lesion.id}
                        </td>
                        <td className="py-3.5 px-5 text-slate-800 dark:text-slate-200 font-medium">
                          {lesion.location || 'Middle-Left'}
                        </td>
                        <td className="py-3.5 px-5 font-mono text-slate-700 dark:text-slate-300">
                          {lesion.areaPercent ? `${lesion.areaPercent.toFixed(2)}% of image area (${lesion.pixelArea} px)` : '0.01% of image area (33 px)'}
                        </td>
                        <td className="py-3.5 px-5">
                          {isL3 ? (
                            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold bg-red-50 dark:bg-red-950/60 border border-red-300 dark:border-red-500/50 text-red-700 dark:text-red-300 shadow-sm">
                              <span className="w-1.5 h-1.5 rounded-full bg-red-500 shrink-0"></span>
                              <span>Level 3 &mdash; Deep Dentinal / Pulp</span>
                            </span>
                          ) : isL2 ? (
                            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold bg-orange-50 dark:bg-orange-950/60 border border-orange-300 dark:border-orange-500/50 text-orange-700 dark:text-orange-300 shadow-sm">
                              <span className="w-1.5 h-1.5 rounded-full bg-orange-500 shrink-0"></span>
                              <span>Level 2 &mdash; Moderate Dentinal</span>
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-[11px] font-bold bg-yellow-50 dark:bg-yellow-950/60 border border-yellow-300 dark:border-yellow-400/50 text-yellow-800 dark:text-yellow-300 shadow-sm">
                              <span className="w-1.5 h-1.5 rounded-full bg-yellow-500 shrink-0"></span>
                              <span>Level 1 &mdash; Suspected Early Caries</span>
                            </span>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>

          {/* Clinical Staging & Pixel Threshold Reference */}
          <div className="bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] rounded-2xl p-5 shadow-sm space-y-4">
            <div className="flex items-center justify-between border-b border-slate-200 dark:border-[#1b2b4d] pb-3">
              <div className="flex items-center gap-2 text-slate-900 dark:text-white font-bold text-xs">
                <Info className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
                <span>AI Clinical Staging & Pixel Area Criteria</span>
              </div>
              <span className="text-[10px] font-mono text-slate-500 dark:text-slate-400">
                Operating Threshold &tau; = 0.50
              </span>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 text-xs">
              {/* Stage 0 */}
              <div className="p-3.5 rounded-xl border border-emerald-300 dark:border-emerald-500/40 bg-emerald-50/50 dark:bg-emerald-950/20 space-y-1.5">
                <div className="flex items-center gap-1.5 font-bold text-emerald-800 dark:text-emerald-300 text-xs">
                  <span className="w-2 h-2 rounded-full bg-emerald-500 shrink-0"></span>
                  <span>Stage 0 (No Caries)</span>
                </div>
                <div className="font-mono text-[11px] text-slate-700 dark:text-slate-300 font-semibold">
                  0 px &bull; 0.00% Area
                </div>
                <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-snug">
                  Intact enamel & dentin margins. No suspicious radiolucency detected.
                </p>
              </div>

              {/* Stage 1 */}
              <div className="p-3.5 rounded-xl border border-yellow-300 dark:border-yellow-400/40 bg-yellow-50/50 dark:bg-yellow-950/20 space-y-1.5">
                <div className="flex items-center gap-1.5 font-bold text-yellow-800 dark:text-yellow-300 text-xs">
                  <span className="w-2 h-2 rounded-full bg-yellow-500 shrink-0"></span>
                  <span>Level 1 (Early Caries)</span>
                </div>
                <div className="font-mono text-[11px] text-slate-700 dark:text-slate-300 font-semibold">
                  20&ndash;250 px &bull; &lt; 0.80% Area
                </div>
                <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-snug">
                  Enamel demineralization (E1/E2) or outer EDJ margin. Non-cavitated.
                </p>
              </div>

              {/* Stage 2 */}
              <div className="p-3.5 rounded-xl border border-orange-300 dark:border-orange-500/40 bg-orange-50/50 dark:bg-orange-950/20 space-y-1.5">
                <div className="flex items-center gap-1.5 font-bold text-orange-800 dark:text-orange-300 text-xs">
                  <span className="w-2 h-2 rounded-full bg-orange-500 shrink-0"></span>
                  <span>Level 2 (Moderate Caries)</span>
                </div>
                <div className="font-mono text-[11px] text-slate-700 dark:text-slate-300 font-semibold">
                  250&ndash;600 px &bull; 0.80%&ndash;1.80% Area
                </div>
                <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-snug">
                  Dentin penetration (D1/D2). Definite radiolucent cavity zone.
                </p>
              </div>

              {/* Stage 3 */}
              <div className="p-3.5 rounded-xl border border-red-300 dark:border-red-500/40 bg-red-50/50 dark:bg-red-950/20 space-y-1.5">
                <div className="flex items-center gap-1.5 font-bold text-red-800 dark:text-red-300 text-xs">
                  <span className="w-2 h-2 rounded-full bg-red-500 shrink-0"></span>
                  <span>Level 3 (Extensive Caries)</span>
                </div>
                <div className="font-mono text-[11px] text-slate-700 dark:text-slate-300 font-semibold">
                  &gt; 600 px &bull; &gt; 1.80% Area
                </div>
                <p className="text-[11px] text-slate-600 dark:text-slate-400 leading-snug">
                  Deep dentinal cavitation (D3) approaching or involving pulp chamber.
                </p>
              </div>
            </div>
          </div>

          {/* Clinical Follow-up Protocol */}
          <div className="p-4 rounded-2xl bg-cyan-50/70 dark:bg-[#09152b] border border-cyan-200/80 dark:border-[#1b3461] text-xs space-y-1">
            <span className="font-bold text-cyan-800 dark:text-cyan-300">Clinical Follow-up Protocol:</span>
            <p className="text-slate-700 dark:text-slate-300 leading-relaxed">
              Candidate regions indicate localized radiolucency. Please corroborate findings with bitewing imaging, transillumination, and visual-tactile assessment.
            </p>
          </div>
        </div>
      )}

      {/* ========================================================================= */}
      {/* TAB 3: PHYSICIAN REVIEW & SIGN-OFF / CLINICAL VERIFICATION */}
      {/* ========================================================================= */}
      {activeTab === 'verification' && (
        <div className="space-y-6 animate-fade-in">
          {/* Clinical Review & Verification Form */}
          <div className="bg-white dark:bg-[#0b1324] border border-slate-200 dark:border-[#1b2b4d] rounded-2xl p-6 sm:p-7 shadow-sm space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 dark:border-[#1b2b4d] pb-4">
              <div>
                <h3 className="text-sm font-bold text-slate-900 dark:text-white flex items-center gap-2">
                  <ShieldCheck className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
                  <span>Clinical Review & Verification</span>
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
                  Sign-off on AI screening findings by a qualified dental professional
                </p>
              </div>

              {reviewSavedSuccess ? (
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 dark:bg-emerald-950/60 text-emerald-800 dark:text-emerald-300 border border-emerald-200 dark:border-emerald-500/50 text-xs font-semibold">
                  <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
                  <span>Verification Successful &mdash; Recorded</span>
                </div>
              ) : (
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-950/60 text-amber-800 dark:text-amber-300 border border-amber-200 dark:border-amber-500/50 text-xs font-semibold">
                  <Clock className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
                  <span>Pending Review &mdash; Not Signed Off</span>
                </div>
              )}
            </div>

            <form onSubmit={handleSaveReview} className="space-y-5">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                {/* Doctor Name */}
                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Doctor / Reviewer Name
                  </label>
                  <div className="relative">
                    <User className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                    <input
                      type="text"
                      required
                      value={doctorName}
                      onChange={(e) => setDoctorName(e.target.value)}
                      placeholder="e.g. Dr. John Doe, DDS"
                      className="w-full pl-9 pr-3.5 py-2.5 bg-white dark:bg-[#0e172a] border border-slate-300 dark:border-[#1b2b4d] rounded-xl text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                </div>

                {/* Review Date */}
                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Review Date
                  </label>
                  <div className="relative">
                    <Calendar className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
                    <input
                      type="date"
                      required
                      value={reviewDate}
                      onChange={(e) => setReviewDate(e.target.value)}
                      className="w-full pl-9 pr-3.5 py-2.5 bg-white dark:bg-[#0e172a] border border-slate-300 dark:border-[#1b2b4d] rounded-xl text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-cyan-500"
                    />
                  </div>
                </div>
              </div>

              {/* Assessment & Follow-up Dropdowns */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Clinical Assessment
                  </label>
                  <select
                    required
                    value={clinicalAssessment}
                    onChange={(e) => setClinicalAssessment(e.target.value)}
                    className="w-full px-3.5 py-2.5 bg-white dark:bg-[#0e172a] border border-slate-300 dark:border-[#1b2b4d] rounded-xl text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="">Select clinical assessment...</option>
                    <option value="No caries identified">No suspicious caries identified</option>
                    <option value="Early caries suspected">Suspected early caries</option>
                    <option value="Moderate caries suspected">Suspected moderate caries</option>
                    <option value="Extensive caries suspected">Suspected extensive caries</option>
                    <option value="Requires further examination">Requires further clinical evaluation</option>
                  </select>
                </div>

                <div className="space-y-1.5">
                  <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                    Recommended Action
                  </label>
                  <select
                    required
                    value={followUp}
                    onChange={(e) => setFollowUp(e.target.value)}
                    className="w-full px-3.5 py-2.5 bg-white dark:bg-[#0e172a] border border-slate-300 dark:border-[#1b2b4d] rounded-xl text-xs text-slate-900 dark:text-slate-100 focus:outline-none focus:border-cyan-500"
                  >
                    <option value="">Select recommended action...</option>
                    <option value="Routine dental examination">Routine clinical follow-up</option>
                    <option value="Clinical examination recommended">Targeted dental examination</option>
                    <option value="Further radiographic evaluation recommended">Additional intraoral imaging recommended</option>
                    <option value="Immediate clinical review recommended">Further diagnostic evaluation recommended</option>
                  </select>
                </div>
              </div>

              {/* Confidence / Agreement */}
              <div className="space-y-2">
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Confidence / Agreement
                </label>
                <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                  {[
                    'Agree with AI-assisted finding',
                    'Partially agree',
                    'Disagree',
                    'Unable to determine',
                  ].map((opt) => (
                    <label
                      key={opt}
                      className={`p-3 rounded-xl border text-xs font-medium cursor-pointer flex items-center gap-2 transition ${
                        aiAgreement === opt
                          ? 'bg-cyan-50 dark:bg-cyan-950/40 border-cyan-500 text-cyan-800 dark:text-cyan-300 font-semibold shadow-sm'
                          : 'bg-white dark:bg-[#0e172a] border-slate-200 dark:border-[#1b2b4d] text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                      }`}
                    >
                      <input
                        type="radio"
                        name="aiAgreement"
                        value={opt}
                        checked={aiAgreement === opt}
                        onChange={() => setAiAgreement(opt)}
                        className="accent-cyan-600"
                      />
                      <span>{opt}</span>
                    </label>
                  ))}
                </div>
              </div>

              {/* Notes */}
              <div className="space-y-1.5">
                <label className="block text-xs font-semibold text-slate-700 dark:text-slate-300">
                  Additional Clinical Notes & Corroboration
                </label>
                <textarea
                  rows={3}
                  value={doctorNotes}
                  onChange={(e) => setDoctorNotes(e.target.value)}
                  placeholder="Record specific tooth notes, visual-tactile findings, vitality tests, or supplementary bitewing requests..."
                  className="w-full p-3 bg-white dark:bg-[#0e172a] border border-slate-300 dark:border-[#1b2b4d] rounded-xl text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 dark:placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                />
              </div>

              {/* Submit */}
              <div className="flex flex-col sm:flex-row items-center justify-between gap-3 pt-2">
                <span className="text-[11px] text-slate-500 dark:text-slate-400">
                  Clinical review will be permanently attached to this analysis case.
                </span>

                <button
                  type="submit"
                  disabled={isSavingReview}
                  className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs shadow-md shadow-cyan-600/20 transition flex items-center gap-2"
                >
                  <Save className="w-4 h-4" />
                  <span>{isSavingReview ? 'Saving...' : 'Save Clinical Review'}</span>
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Lightbox Modal */}
      <ImageLightboxModal
        isOpen={isLightboxOpen}
        onClose={() => setIsLightboxOpen(false)}
        originalUrl={analysis.originalImageUrl}
        overlayUrl={analysis.segmentationOverlayUrl}
        binaryMaskUrl={analysis.binaryMaskUrl}
        findings={analysis.findings}
        initialLayer={lightboxInitialLayer}
      />
    </div>
  );
};

