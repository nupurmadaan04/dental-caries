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
  Award,
} from 'lucide-react';
import { PRODUCT_INFO, MODEL_BENCHMARK_METRICS } from '../constants/clinicalMetadata';

export const TechnicalVerificationPage: React.FC = () => {
  const validationMetrics = [
    {
      name: 'Dice Similarity Coefficient',
      percent: MODEL_BENCHMARK_METRICS.validation.dicePercent,
      raw: MODEL_BENCHMARK_METRICS.validation.dice.toFixed(5),
      desc: `Pixel-level spatial overlap on validation cohort (Epoch ${MODEL_BENCHMARK_METRICS.selectedEpoch}, ${MODEL_BENCHMARK_METRICS.operatingThresholdStr})`
    },
    {
      name: 'Intersection over Union (IoU)',
      percent: MODEL_BENCHMARK_METRICS.validation.iouPercent,
      raw: MODEL_BENCHMARK_METRICS.validation.iou.toFixed(5),
      desc: 'Jaccard similarity index across foreground caries candidate regions'
    },
    {
      name: 'Precision (PPV)',
      percent: MODEL_BENCHMARK_METRICS.validation.precisionPercent,
      raw: MODEL_BENCHMARK_METRICS.validation.precision.toFixed(5),
      desc: 'Proportion of true positive caries pixels among all predicted positives'
    },
    {
      name: 'Recall / Sensitivity',
      percent: MODEL_BENCHMARK_METRICS.validation.recallPercent,
      raw: MODEL_BENCHMARK_METRICS.validation.recall.toFixed(5),
      desc: 'True positive caries detection rate across ground-truth lesions'
    },
    {
      name: 'Specificity (TNR)',
      percent: MODEL_BENCHMARK_METRICS.validation.specificityPercent,
      raw: MODEL_BENCHMARK_METRICS.validation.specificity.toFixed(5),
      desc: 'True negative detection rate on healthy background dental tissues'
    },
    {
      name: 'Validation Loss',
      percent: MODEL_BENCHMARK_METRICS.validation.loss.toString(),
      raw: MODEL_BENCHMARK_METRICS.validation.loss.toString(),
      desc: `Combined BCE + Dice Loss at peak validation performance (Epoch ${MODEL_BENCHMARK_METRICS.selectedEpoch})`
    },
  ];

  const sealedTestMetrics = [
    {
      name: 'Macro Dice',
      percent: MODEL_BENCHMARK_METRICS.sealedTest.macroDicePercent,
      raw: MODEL_BENCHMARK_METRICS.sealedTest.macroDice.toFixed(5),
      desc: 'Average Dice score across 100 untouched test cases (evaluating unweighted case performance)'
    },
    {
      name: 'Macro IoU',
      percent: MODEL_BENCHMARK_METRICS.sealedTest.macroIouPercent,
      raw: MODEL_BENCHMARK_METRICS.sealedTest.macroIou.toFixed(5),
      desc: 'Average Jaccard index across all 100 test cases'
    },
    {
      name: 'Macro Precision',
      percent: MODEL_BENCHMARK_METRICS.sealedTest.macroPrecisionPercent,
      raw: MODEL_BENCHMARK_METRICS.sealedTest.macroPrecision.toFixed(5),
      desc: 'Average case-level precision'
    },
    {
      name: 'Macro Recall',
      percent: MODEL_BENCHMARK_METRICS.sealedTest.macroRecallPercent,
      raw: MODEL_BENCHMARK_METRICS.sealedTest.macroRecall.toFixed(5),
      desc: 'Average case-level lesion sensitivity'
    },
    {
      name: 'Macro Specificity',
      percent: MODEL_BENCHMARK_METRICS.sealedTest.macroSpecificityPercent,
      raw: MODEL_BENCHMARK_METRICS.sealedTest.macroSpecificity.toFixed(5),
      desc: 'High background specificity maintained across diverse radiographs'
    },
    {
      name: 'Micro Dice / F1',
      percent: MODEL_BENCHMARK_METRICS.sealedTest.microDicePercent,
      raw: MODEL_BENCHMARK_METRICS.sealedTest.microDice.toFixed(5),
      desc: 'Global pixel-level Dice calculated across all 117,964,800 test pixels'
    },
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
              CURRENT VALIDATION-BEST CHECKPOINT: {MODEL_BENCHMARK_METRICS.modelName}
            </div>
            <div className="text-slate-600 dark:text-slate-400 text-[11px] font-mono mt-0.5">
              Checkpoint: <span className="font-semibold text-slate-800 dark:text-slate-200">{PRODUCT_INFO.selectedCheckpoint}</span> | Threshold: <span className="font-semibold text-cyan-600 dark:text-cyan-400">{PRODUCT_INFO.operatingThreshold}</span>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="px-2.5 py-1 rounded-full bg-emerald-50 dark:bg-emerald-500/10 border border-emerald-300 dark:border-emerald-500/30 text-emerald-700 dark:text-emerald-400 font-mono font-semibold text-[11px]">
            Validation Best: Epoch {MODEL_BENCHMARK_METRICS.selectedEpoch} | Run completed: Epoch {MODEL_BENCHMARK_METRICS.totalTrainingEpochs}
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
            <span className="text-3xl font-black text-cyan-600 dark:text-cyan-400 font-mono">{MODEL_BENCHMARK_METRICS.validation.dicePercent}</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">{MODEL_BENCHMARK_METRICS.validation.dice}</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Peak validation overlap (Epoch {MODEL_BENCHMARK_METRICS.selectedEpoch})</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider block">
            IoU (Jaccard)
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-cyan-600 dark:text-cyan-400 font-mono">{MODEL_BENCHMARK_METRICS.validation.iouPercent}</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">{MODEL_BENCHMARK_METRICS.validation.iou}</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Foreground area overlap</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider block">
            Validation Precision
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-emerald-600 dark:text-emerald-400 font-mono">{MODEL_BENCHMARK_METRICS.validation.precisionPercent}</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">{MODEL_BENCHMARK_METRICS.validation.precision}</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">True positive pixel ratio</p>
        </div>

        <div className="p-5 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-sm">
          <span className="text-[11px] font-semibold text-slate-600 dark:text-slate-400 uppercase tracking-wider block">
            Validation Recall
          </span>
          <div className="flex items-baseline gap-2 mt-1">
            <span className="text-3xl font-black text-cyan-600 dark:text-cyan-400 font-mono">{MODEL_BENCHMARK_METRICS.validation.recallPercent}</span>
            <span className="text-xs text-slate-500 dark:text-slate-400 font-mono">{MODEL_BENCHMARK_METRICS.validation.recall}</span>
          </div>
          <p className="text-[10px] text-slate-500 dark:text-slate-400 mt-1">Demineralization detection rate</p>
        </div>
      </div>

      {/* Literature Comparison Callout */}
      <div className="p-6 rounded-3xl bg-gradient-to-r from-cyan-500/10 via-teal-500/10 to-transparent border border-cyan-500/20 shadow-sm flex flex-col md:flex-row items-start md:items-center justify-between gap-4">
        <div className="space-y-1">
          <div className="flex items-center gap-2">
            <Award className="w-5 h-5 text-cyan-500" />
            <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100">
              Literature Benchmark Comparison
            </h3>
          </div>
          <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed max-w-2xl">
            Validation Dice: <span className="font-semibold text-cyan-600 dark:text-cyan-400">71.867%</span>. This is <span className="font-semibold text-emerald-600 dark:text-emerald-400">+0.747 percentage points</span> above the <span className="font-semibold text-slate-800 dark:text-slate-200">71.12%</span> literature reference under this project's fixed validation protocol.
          </p>
          <p className="text-[11px] text-slate-500 dark:text-slate-400 italic">
            Note: Research benchmark comparison only. Does not imply autonomous clinical superiority or statistical generalization guarantees.
          </p>
        </div>
        <div className="flex items-center gap-4 bg-white/70 dark:bg-[#0c1424]/70 px-4 py-2.5 rounded-2xl border border-cyan-500/30">
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">Lit. Ref</div>
            <div className="text-lg font-black font-mono text-slate-700 dark:text-slate-300">71.12%</div>
          </div>
          <div className="h-8 w-px bg-slate-200 dark:bg-slate-700" />
          <div>
            <div className="text-[10px] uppercase tracking-wider text-slate-500 font-semibold">E75 Validation</div>
            <div className="text-lg font-black font-mono text-cyan-600 dark:text-cyan-400">71.87%</div>
          </div>
          <div className="h-8 w-px bg-slate-200 dark:bg-slate-700" />
          <div>
            <div className="text-[10px] uppercase tracking-wider text-emerald-600 font-semibold">Margin</div>
            <div className="text-lg font-black font-mono text-emerald-600 dark:text-emerald-400">+0.747 pp</div>
          </div>
        </div>
      </div>

      {/* Validation Set Metrics Table */}
      <div className="bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] rounded-3xl p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <Activity className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
              <span>Model Validation Performance ({MODEL_BENCHMARK_METRICS.experimentId}, Epoch {MODEL_BENCHMARK_METRICS.selectedEpoch})</span>
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
              Evaluated on canonical validation dental radiograph cohort at threshold {MODEL_BENCHMARK_METRICS.operatingThresholdStr}
            </p>
          </div>
          <span className="text-[11px] font-mono px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 border border-cyan-200 dark:border-cyan-500/20 font-semibold">
            Selected Operating Point: {MODEL_BENCHMARK_METRICS.operatingThresholdStr}
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

      {/* FINAL SEALED-TEST EVALUATION SECTION */}
      <div className="bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] rounded-3xl p-6 sm:p-8 shadow-sm space-y-6">
        <div className="flex items-center justify-between flex-wrap gap-2">
          <div>
            <div className="inline-flex items-center gap-2 px-2.5 py-0.5 rounded-full bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 text-amber-700 dark:text-amber-400 text-[10px] font-semibold mb-1">
              <Lock className="w-3 h-3" />
              <span>Independent Evaluation Cohort</span>
            </div>
            <h2 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <ShieldCheck className="w-5 h-5 text-emerald-600 dark:text-emerald-400" />
              <span>Final Sealed-Test Evaluation</span>
            </h2>
            <p className="text-xs text-slate-600 dark:text-slate-400 mt-0.5">
              Checkpoint: <span className="font-mono font-semibold">{MODEL_BENCHMARK_METRICS.checkpoint}</span> | Threshold: <span className="font-mono font-semibold">{MODEL_BENCHMARK_METRICS.operatingThresholdStr}</span> | Cases: <span className="font-semibold">{MODEL_BENCHMARK_METRICS.sealedTest.totalCases}</span>
            </p>
          </div>
          <div className="text-right">
            <span className="px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 font-mono text-[11px] font-semibold border border-slate-200 dark:border-slate-700">
              Evaluated once on the untouched sealed test set
            </span>
          </div>
        </div>

        <div className="p-4 rounded-2xl bg-slate-50 dark:bg-[#080d1a] border border-slate-200 dark:border-[#1b2b4d] text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
          <span className="font-semibold text-slate-800 dark:text-slate-200">Protocol Transparency: </span>
          Evaluated once on the untouched sealed test set. The validation-to-test Dice gap is <span className="font-mono font-semibold text-slate-700 dark:text-slate-300">-21.72 percentage points</span> (Macro Test Dice: 50.15% vs Validation Dice: 71.87%). This reflects real-world clinical dataset shift, subtle lesion boundary variability, and unaugmented distribution differences. Labeled strictly as <span className="font-semibold text-slate-800 dark:text-slate-200">Final Sealed-Test Evaluation</span> (not clinical validation or production guarantee).
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-[#080d1a] border-b border-slate-200 dark:border-[#1b2b4d] text-slate-600 dark:text-slate-400 font-semibold uppercase text-[10px]">
                <th className="py-3 px-4">Test Metric</th>
                <th className="py-3 px-4">Score (%)</th>
                <th className="py-3 px-4">Description</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-[#1b2b4d]">
              {sealedTestMetrics.map((m, idx) => (
                <tr key={idx} className="hover:bg-slate-50 dark:hover:bg-[#121b2d]/50 transition">
                  <td className="py-3.5 px-4 font-semibold text-slate-900 dark:text-slate-100">
                    {m.name}
                  </td>
                  <td className="py-3.5 px-4">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-black text-emerald-600 dark:text-emerald-400 text-sm">
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
