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
  Lock,
  Cpu,
  Layers,
  Sparkles,
} from 'lucide-react';
import { PRODUCT_INFO, MODEL_BENCHMARK_METRICS } from '../constants/clinicalMetadata';

export const TechnicalVerificationPage: React.FC = () => {
  const validationMetrics = [
    { name: 'Dice Similarity Coefficient', percent: '65.62%', raw: '0.6562', desc: 'Pixel-level spatial overlap on validation cohort (E56, τ=0.50)' },
    { name: 'Intersection over Union (IoU)', percent: '49.85%', raw: '0.4985', desc: 'Jaccard similarity index across foreground caries regions' },
    { name: 'Precision (PPV)', percent: '69.01%', raw: '0.6901', desc: 'Proportion of true positive caries pixels among all predicted positives' },
    { name: 'Recall / Sensitivity', percent: '63.65%', raw: '0.6365', desc: 'True positive caries detection rate across ground-truth lesions' },
    { name: 'Validation Loss', percent: '0.7639', raw: '0.7639', desc: 'Combined BCE + Dice Loss at peak validation performance' },
  ];

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12 animate-fade-in text-slate-900 dark:text-slate-100">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 text-xs font-semibold mb-2">
          <ShieldCheck className="w-3.5 h-3.5" />
          <span>Verified Research Benchmark & Validation Metrics</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
          Evaluation & Verification
        </h1>
        <p className="text-slate-600 dark:text-slate-400 text-sm mt-1">
          Validated benchmark metrics, segmentation standards, and technical specifications for {PRODUCT_INFO.name}.
        </p>
      </div>

      {/* Model Spec Callout */}
      <div className="p-5 rounded-2xl bg-white dark:bg-[#0c1424] border border-cyan-200 dark:border-cyan-500/30 text-xs flex flex-wrap items-center justify-between gap-4 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-xl bg-cyan-50 dark:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400 border border-cyan-200 dark:border-cyan-500/30">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="font-bold text-slate-900 dark:text-slate-100 text-sm">
              {MODEL_BENCHMARK_METRICS.modelName}
            </div>
            <div className="text-slate-600 dark:text-slate-400 text-[11px] font-mono mt-0.5">
              Checkpoint: <span className="font-semibold text-slate-800 dark:text-slate-200">{PRODUCT_INFO.selectedCheckpoint}</span> | Operating Point: <span className="font-semibold text-cyan-600 dark:text-cyan-400">{PRODUCT_INFO.operatingThreshold}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-300 dark:border-emerald-500/30 text-emerald-700 dark:text-emerald-400 font-mono font-semibold text-[11px]">
            Training Complete (60 Epochs)
          </span>
          <span className="px-2.5 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-300 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 font-mono font-semibold text-[11px]">
            Research Frozen
          </span>
        </div>
      </div>

      {/* Top Metric Cards - Real Validation Performance */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        <div className="p-5 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider block">
            Validation Dice
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-cyan-600 dark:text-cyan-400 font-mono">65.62%</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">0.6562</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Peak validation overlap (E56)</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider block">
            IoU (Jaccard)
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-cyan-600 dark:text-cyan-400 font-mono">49.85%</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">0.4985</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Foreground area overlap</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider block">
            Validation Precision
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-emerald-600 dark:text-emerald-400 font-mono">69.01%</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">0.6901</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">True positive pixel ratio</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider block">
            Validation Recall
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-cyan-600 dark:text-cyan-400 font-mono">63.65%</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">0.6365</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Demineralization detection rate</p>
        </div>
      </div>

      {/* Validation Set Metrics Table */}
      <div className="bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] rounded-3xl p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
              <span>Model Validation Performance (EXP-MLUA-003, Epoch 56)</span>
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
              Evaluated on canonical validation dental radiograph cohort at threshold τ = 0.50
            </p>
          </div>
          <span className="text-[11px] font-mono px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 border border-cyan-200 dark:border-cyan-500/20 font-semibold">
            Selected Operating Point: τ = 0.50
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-[#080d1a] border-b border-slate-200 dark:border-[#1b2b4d] text-slate-600 dark:text-slate-400 font-semibold uppercase text-[10px]">
                <th className="py-3 px-4">Metric</th>
                <th className="py-3 px-4">Score (%)</th>
                <th className="py-3 px-4">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-[#1b2b4d]">
              {validationMetrics.map((m, idx) => (
                <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-[#121b2d]/50 transition">
                  <td className="py-3.5 px-4 font-semibold text-slate-900 dark:text-slate-100">
                    {m.name}
                  </td>
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-black text-cyan-600 dark:text-cyan-400 text-sm">
                        {m.percent}
                      </span>
                      <span className="font-mono text-[11px] text-slate-500 dark:text-slate-400">
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
        <div className="p-6 rounded-3xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 dark:text-slate-100 font-bold text-sm">
            <Database className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
            <span>DC1000 Dataset & Patch Infrastructure</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            Trained and evaluated on the DC1000 dental caries panoramic cohort (1,000 cases; 2,389 training patches at 20% active supervision rate DICE530). Uses 21-patch sliding-window inference with spatial overlap reconstruction.
          </p>
        </div>

        <div className="p-6 rounded-3xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm space-y-3">
          <div className="flex items-center gap-2 text-slate-900 dark:text-slate-100 font-bold text-sm">
            <Lock className="w-4 h-4 text-emerald-600 dark:text-emerald-400" />
            <span>Sealed Test Protocol & Protection</span>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
            The 100-case test set was strictly sealed and evaluated once after freezing the model and threshold at τ = 0.50. Zero test data was used for training, checkpoint selection, or threshold sweeping.
          </p>
        </div>
      </div>
    </div>
  );
};
