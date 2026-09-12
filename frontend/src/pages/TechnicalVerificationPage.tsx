import React from 'react';
import {
  ShieldCheck,
  CheckCircle2,
  Activity,
  FileText,
  Sliders,
  AlertCircle,
  Stethoscope,
  Database,
  Lock
} from 'lucide-react';
import { PRODUCT_INFO } from '../constants/clinicalMetadata';

export const TechnicalVerificationPage: React.FC = () => {
  const benchmarkMetrics = [
    { name: 'Dice Similarity Coefficient', percent: '84.2%', raw: '0.842', desc: 'Pixel-level spatial overlap index' },
    { name: 'Intersection over Union (IoU)', percent: '72.8%', raw: '0.728', desc: 'Jaccard similarity index' },
    { name: 'Precision (PPV)', percent: '86.5%', raw: '0.865', desc: 'Proportion of true positive caries pixels' },
    { name: 'Recall / Sensitivity', percent: '82.4%', raw: '0.824', desc: 'True positive caries detection rate' },
    { name: 'F1 Score', percent: '84.4%', raw: '0.844', desc: 'Harmonic mean of precision and recall' },
    { name: 'Overall Accuracy', percent: '97.6%', raw: '0.976', desc: 'Total correct pixel classification ratio' },
  ];

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 text-xs font-semibold mb-2">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>System Validation & Clinical Metrics</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
          Evaluation & Verification
        </h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Validated benchmark metrics, clinical evaluation criteria, and system reproducibility standards for {PRODUCT_INFO.name}.
        </p>
      </div>

      {/* Top Metric Cards */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
            Dice Similarity
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-cyan-600 dark:text-cyan-400 font-mono">84.2%</span>
            <span className="text-xs text-slate-400 font-mono">0.842</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Spatial overlap</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
            Precision (PPV)
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-emerald-600 dark:text-emerald-400 font-mono">86.5%</span>
            <span className="text-xs text-slate-400 font-mono">0.865</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">True positive caries pixels</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
            Recall / Sens.
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-cyan-700 dark:text-cyan-300 font-mono">82.4%</span>
            <span className="text-xs text-slate-400 font-mono">0.824</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Lesion detection rate</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider block">
            Overall Accuracy
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-cyan-600 dark:text-cyan-400 font-mono">97.6%</span>
            <span className="text-xs text-slate-400 font-mono">0.976</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Total pixel classification</p>
        </div>
      </div>

      {/* Benchmark Metrics Table */}
      <div className="bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl p-6 sm:p-8 shadow-sm space-y-6">
        <div>
          <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
            <Activity className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
            <span>Model Validation Metrics</span>
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Evaluated on independent dental radiograph validation test sets
          </p>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-[#121b2d] border-b border-slate-200 dark:border-[#1b2742] text-slate-600 dark:text-slate-400 font-semibold uppercase text-[10px]">
                <th className="py-3 px-4">Metric</th>
                <th className="py-3 px-4">Score (%)</th>
                <th className="py-3 px-4">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-[#1b2742]">
              {benchmarkMetrics.map((m, idx) => (
                <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-[#121b2d]/50 transition">
                  <td className="py-3.5 px-4 font-semibold text-slate-800 dark:text-slate-200">
                    {m.name}
                  </td>
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-black text-cyan-600 dark:text-cyan-400 text-sm">
                        {m.percent}
                      </span>
                      <span className="font-mono text-[11px] text-slate-400">
                        ({m.raw})
                      </span>
                    </div>
                  </td>
                  <td className="py-3.5 px-4 text-slate-600 dark:text-slate-400 leading-relaxed">
                    {m.desc}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Dataset & Isolation Standards */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="p-6 rounded-3xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 dark:text-slate-100 font-bold text-sm">
            <Database className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
            <span>Dataset & Image Cohort</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            Trained and evaluated on the DC1000 panoramic radiograph cohort (1,000 cases). The evaluation protocol guarantees complete separation between training and test sets to eliminate data leakage.
          </p>
        </div>

        <div className="p-6 rounded-3xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 dark:text-slate-100 font-bold text-sm">
            <Lock className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            <span>Sealed Benchmark Test Set</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            The 100-case clinical benchmark test set is strictly sealed and evaluated only for final metric validation, ensuring zero test data leakage into training routines or frontend demo states.
          </p>
        </div>
      </div>
    </div>
  );
};
