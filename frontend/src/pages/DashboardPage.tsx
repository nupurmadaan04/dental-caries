import React, { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  UploadCloud,
  FileText,
  Activity,
  CheckCircle2,
  AlertTriangle,
  Stethoscope,
  ShieldCheck,
  Clock,
  ArrowRight,
  Sparkles,
  Layers,
  ChevronRight,
  FolderOpen
} from 'lucide-react';
import { MetricCard } from '../components/MetricCard';
import { PRODUCT_INFO, CLINICAL_STAGES } from '../constants/clinicalMetadata';
import { apiService } from '../services/api';
import { HistoryItem } from '../types/api';

export const DashboardPage: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [reportsCount, setReportsCount] = useState<number>(0);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    Promise.all([apiService.getHistory(), apiService.getReports()]).then(([hist, reps]) => {
      setHistory(hist);
      setReportsCount(reps.length);
      setLoading(false);
    });
  }, []);

  const totalAnalyzed = history.length;
  const potentialCaries = history.filter((h) => {
    const isNoCaries = 
      h.lesionCount === 0 || 
      h.stageLevel?.toLowerCase().includes('0') || 
      h.overallFinding?.toLowerCase().includes('no caries') ||
      (h as any).finding?.toLowerCase().includes('no caries') ||
      h.stageLevel?.toLowerCase().includes('no significant');
    return !isNoCaries && (
      (typeof h.lesionCount === 'number' && h.lesionCount > 0) ||
      (h.overallFinding && !h.overallFinding.toLowerCase().includes('no caries'))
    );
  }).length;
  const casesRequiringReview = history.filter((h) => !h.isReviewed).length;
  const cariesRate = totalAnalyzed > 0 ? ((potentialCaries / totalAnalyzed) * 100).toFixed(1) : '0.0';

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Clinical Hero Section - Full Light & Dark Theme Support */}
      <div className="relative rounded-3xl overflow-hidden bg-white dark:bg-gradient-to-r dark:from-slate-900 dark:via-[#0d1322] dark:to-[#121b2d] border border-slate-200 dark:border-[#1b2742] p-6 sm:p-10 shadow-sm dark:shadow-xl text-slate-900 dark:text-white">
        <div className="absolute top-0 right-0 -mt-8 -mr-8 w-96 h-96 bg-cyan-500/10 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/20 border border-cyan-200 dark:border-cyan-400/30 text-cyan-800 dark:text-cyan-300 text-xs font-semibold">
            <Activity className="w-3.5 h-3.5 text-cyan-600 dark:text-cyan-300" />
            <span>{PRODUCT_INFO.modality}</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-extrabold tracking-tight text-slate-900 dark:text-white leading-tight">
            {PRODUCT_INFO.name}
          </h1>

          <p className="text-slate-600 dark:text-slate-300 text-sm sm:text-base leading-relaxed">
            AI-assisted analysis of panoramic dental radiographs for caries identification, pixel-level candidate localization, and clinical decision support.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <Link
              to="/analyze"
              className="flex items-center gap-2 px-5 py-3 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-sm shadow-md shadow-cyan-600/20 transition-all hover:scale-[1.02]"
            >
              <UploadCloud className="w-4 h-4" />
              <span>Analyze Radiograph</span>
            </Link>

            <Link
              to="/methodology"
              className="flex items-center gap-2 px-5 py-3 rounded-xl bg-slate-100 dark:bg-slate-800/80 hover:bg-slate-200 dark:hover:bg-slate-700/80 border border-slate-200 dark:border-slate-700 text-slate-700 dark:text-slate-200 font-semibold text-sm transition"
            >
              <FileText className="w-4 h-4" />
              <span>View Methods</span>
            </Link>
          </div>
        </div>
      </div>

      {/* Clinical KPI Grid - Real Dynamic Values */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 sm:gap-6">
        <MetricCard
          title="Radiographs Analyzed"
          value={loading ? '...' : totalAnalyzed.toString()}
          subtitle="Panoramic screening cases"
          icon={Activity}
          trend={`${totalAnalyzed} total cases`}
          badge="Active"
          badgeType="brand"
        />

        <MetricCard
          title="Potential Caries Cases"
          value={loading ? '...' : potentialCaries.toString()}
          subtitle="Screening candidate positive"
          icon={AlertTriangle}
          trend={`${cariesRate}% screening rate`}
          badge="Candidate"
          badgeType="warning"
        />

        <MetricCard
          title="Cases Requiring Review"
          value={loading ? '...' : casesRequiringReview.toString()}
          subtitle="Pending physician sign-off"
          icon={Clock}
          badge={casesRequiringReview > 0 ? "Action Required" : "All Reviewed"}
          badgeType={casesRequiringReview > 0 ? "purple" : "success"}
        />

        <MetricCard
          title="Reports Generated"
          value={loading ? '...' : reportsCount.toString()}
          subtitle="Clinical AI summaries archived"
          icon={FileText}
          trend="PDF & JSON exports"
          badge="Archived"
          badgeType="success"
        />
      </div>

      {/* Clinical Screening Severity Guide */}
      <div className="bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl p-6 sm:p-8 shadow-sm dark:shadow-xl space-y-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-200 dark:border-[#1b2742] pb-4">
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Stethoscope className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
              <span>Clinical Caries Screening Categories</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Standardized area-based severity definitions for panoramic radiograph decision support
            </p>
          </div>
          <Link
            to="/limitations"
            className="text-xs font-semibold text-cyan-600 dark:text-cyan-400 hover:underline flex items-center gap-1 self-start sm:self-auto"
          >
            <span>Review Limitations</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {Object.values(CLINICAL_STAGES).map((stg) => (
            <div
              key={stg.key}
              className={`p-4 rounded-2xl border transition-all ${
                stg.key === 'STAGE_3'
                  ? 'bg-rose-50/50 dark:bg-rose-950/20 border-rose-200 dark:border-rose-900/40'
                  : stg.key === 'STAGE_2'
                  ? 'bg-amber-50/50 dark:bg-amber-950/20 border-amber-200 dark:border-amber-900/40'
                  : stg.key === 'STAGE_1'
                  ? 'bg-cyan-50/50 dark:bg-cyan-950/20 border-cyan-200 dark:border-cyan-900/40'
                  : 'bg-emerald-50/50 dark:bg-emerald-950/20 border-emerald-200 dark:border-emerald-900/40'
              }`}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-bold font-mono uppercase px-2 py-0.5 rounded-full bg-white dark:bg-[#121b2d] border border-current">
                  {stg.label}
                </span>
                <span className="text-[10px] font-semibold text-slate-500 dark:text-slate-400">
                  {stg.severity}
                </span>
              </div>
              <h4 className="font-bold text-slate-900 dark:text-slate-100 text-sm mb-1">{stg.title}</h4>
              <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed mb-3">{stg.description}</p>
              <div className="text-[11px] font-medium text-slate-700 dark:text-slate-300 bg-white/80 dark:bg-slate-900/60 p-2.5 rounded-xl border border-slate-200 dark:border-slate-800">
                {stg.recommendation}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Recent Radiograph Cases */}
      <div className="bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl p-6 sm:p-8 shadow-sm dark:shadow-xl space-y-6">
        <div className="flex items-center justify-between border-b border-slate-200 dark:border-[#1b2742] pb-4">
          <div>
            <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <FolderOpen className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
              <span>Recent Radiograph Cases</span>
            </h2>
            <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
              Latest panoramic scans processed for clinical decision support
            </p>
          </div>
          <Link
            to="/history"
            className="text-xs font-semibold text-cyan-600 dark:text-cyan-400 hover:underline flex items-center gap-1"
          >
            <span>View All History</span>
            <ChevronRight className="w-3.5 h-3.5" />
          </Link>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-[#121b2d] text-slate-500 dark:text-slate-400 border-b border-slate-200 dark:border-[#1b2742] uppercase text-[10px] font-semibold">
                <th className="py-3 px-4">Case ID</th>
                <th className="py-3 px-4">Radiograph File</th>
                <th className="py-3 px-4">Exam Date</th>
                <th className="py-3 px-4 text-center">Stage</th>
                <th className="py-3 px-4">Lesions</th>
                <th className="py-3 px-4">Review Status</th>
                <th className="py-3 px-4 text-right">Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-[#1b2742]">
              {history.slice(0, 5).map((item) => {
                const isNoCaries = 
                  item.lesionCount === 0 || 
                  item.stageLevel === '0' ||
                  item.stageLevel?.toLowerCase().includes('0') || 
                  item.overallFinding?.toLowerCase().includes('no caries') ||
                  (item as any).finding?.toLowerCase().includes('no caries') ||
                  item.stageLevel?.toLowerCase().includes('no significant');

                const stageDigit = isNoCaries
                  ? '0'
                  : item.stageLevel?.includes('3') || item.overallFinding?.toLowerCase().includes('extensive')
                  ? '3'
                  : item.stageLevel?.includes('2') || item.overallFinding?.toLowerCase().includes('moderate')
                  ? '2'
                  : '1';

                const lesionNum = isNoCaries ? 0 : (typeof item.lesionCount === 'number' ? item.lesionCount : ((item as any).lesions ?? 1));

                const badgeStyle = isNoCaries
                  ? 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-800 dark:text-emerald-300 border-emerald-200 dark:border-emerald-500/30'
                  : stageDigit === '3'
                  ? 'bg-red-50 dark:bg-red-500/10 text-red-800 dark:text-red-300 border-red-200 dark:border-red-500/30'
                  : stageDigit === '2'
                  ? 'bg-orange-50 dark:bg-orange-500/10 text-orange-800 dark:text-orange-300 border-orange-200 dark:border-orange-500/30'
                  : 'bg-yellow-50 dark:bg-yellow-500/10 text-yellow-800 dark:text-yellow-300 border-yellow-200 dark:border-yellow-500/30';

                return (
                  <tr key={item.id} className="hover:bg-slate-50 dark:hover:bg-[#121b2d]/50 transition">
                    <td className="py-3.5 px-4 font-mono font-bold text-cyan-600 dark:text-cyan-400">
                      {item.id}
                    </td>
                    <td className="py-3.5 px-4 font-medium text-slate-800 dark:text-slate-200">
                      <div className="max-w-[150px] truncate" title={item.filename}>
                        {item.filename}
                      </div>
                    </td>
                    <td className="py-3.5 px-4 text-slate-500 dark:text-slate-400 font-mono text-[11px]">
                      {item.timestamp}
                    </td>
                    <td className="py-3.5 px-4 text-center">
                      <span className={`inline-flex items-center justify-center px-2.5 py-1 rounded-full text-xs font-bold font-mono border ${badgeStyle}`}>
                        Stage {stageDigit}
                      </span>
                    </td>
                    <td className="py-3.5 px-4 font-mono font-bold text-slate-900 dark:text-slate-100">
                      {lesionNum} {lesionNum === 1 ? 'site' : 'sites'}
                    </td>
                    <td className="py-3.5 px-4">
                      {item.isReviewed ? (
                        <span className="inline-flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400 font-bold">
                          <CheckCircle2 className="w-3.5 h-3.5" />
                          Verified
                        </span>
                      ) : (
                        <Link
                          to={`/results/${item.id}?tab=verification`}
                          className="inline-flex items-center gap-1 text-[11px] text-amber-600 dark:text-amber-400 hover:text-amber-500 hover:underline font-bold transition group cursor-pointer"
                          title="Click to open Physician Review & Sign-Off"
                        >
                          <Clock className="w-3.5 h-3.5 group-hover:scale-110 transition-transform" />
                          <span>Pending</span>
                        </Link>
                      )}
                    </td>
                    <td className="py-3.5 px-4 text-right">
                      <Link
                        to={`/results/${item.id}`}
                        className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs transition shadow-sm"
                      >
                        <span>Review</span>
                        <ArrowRight className="w-3 h-3" />
                      </Link>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
