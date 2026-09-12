import React from 'react';
import { LesionFinding } from '../types/api';
import { Stethoscope, CheckCircle2, AlertCircle, Eye } from 'lucide-react';

interface LesionTableProps {
  findings: LesionFinding[];
  selectedId?: string;
  onSelectFinding?: (id: string) => void;
}

export const LesionTable: React.FC<LesionTableProps> = ({
  findings,
  selectedId,
  onSelectFinding,
}) => {
  if (!findings || findings.length === 0) {
    return (
      <div className="bg-white dark:bg-[#121b2d]/80 border border-slate-200 dark:border-[#1b2742] rounded-2xl p-8 text-center space-y-2">
        <CheckCircle2 className="w-8 h-8 text-emerald-500 mx-auto" />
        <h4 className="text-sm font-bold text-slate-800 dark:text-slate-100">No Caries Candidates Identified</h4>
        <p className="text-xs text-slate-500 dark:text-slate-400 max-w-md mx-auto">
          Automated pixel segmentation did not detect radiolucent lesion candidates exceeding the threshold. Routine prophylactic care recommended.
        </p>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-[#121b2d]/80 border border-slate-200 dark:border-[#1b2742] rounded-2xl overflow-hidden shadow-md">
      <div className="p-4 border-b border-slate-200 dark:border-[#1b2742] flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Stethoscope className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
          <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">
            Detected Lesion Candidate Sites ({findings.length})
          </h3>
        </div>
        <span className="text-[11px] text-slate-500 dark:text-slate-400">
          Click row to highlight localization
        </span>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-left border-collapse text-xs">
          <thead>
            <tr className="bg-slate-50 dark:bg-[#0d1322] border-b border-slate-200 dark:border-[#1b2742] text-slate-600 dark:text-slate-400 font-semibold uppercase tracking-wider text-[10px]">
              <th className="py-3 px-4">Lesion</th>
              <th className="py-3 px-4">Location</th>
              <th className="py-3 px-4">Area</th>
              <th className="py-3 px-4">Area %</th>
              <th className="py-3 px-4">Confidence</th>
              <th className="py-3 px-4">Stage</th>
              <th className="py-3 px-4">Clinical Review Guidance</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-100 dark:divide-[#1b2742]">
            {findings.map((lesion) => {
              const isSelected = selectedId === lesion.id;
              return (
                <tr
                  key={lesion.id}
                  onClick={() => onSelectFinding && onSelectFinding(lesion.id)}
                  className={`cursor-pointer transition-colors ${
                    isSelected
                      ? 'bg-cyan-50 dark:bg-cyan-950/40 border-l-4 border-cyan-500'
                      : 'hover:bg-slate-50 dark:hover:bg-[#1a243c]/60'
                  }`}
                >
                  <td className="py-3 px-4 font-bold text-cyan-700 dark:text-cyan-400 flex items-center gap-1.5">
                    <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
                    {lesion.lesionNumber || lesion.id}
                  </td>
                  <td className="py-3 px-4 text-slate-800 dark:text-slate-200 font-medium">
                    {lesion.location} {lesion.toothNumberFDI ? `(FDI #${lesion.toothNumberFDI})` : ''}
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-700 dark:text-slate-300">
                    {lesion.pixelArea} px
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-700 dark:text-slate-300">
                    {lesion.areaPercent ? `${lesion.areaPercent.toFixed(2)}%` : '< 0.1%'}
                  </td>
                  <td className="py-3 px-4 font-mono text-slate-800 dark:text-slate-200 font-semibold">
                    {(lesion.confidence * 100).toFixed(0)}%
                  </td>
                  <td className="py-3 px-4">
                    <span className={`px-2 py-0.5 rounded-full text-[10px] font-semibold border ${
                      lesion.stage === 'Stage 3'
                        ? 'bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border-rose-300 dark:border-rose-500/30'
                        : lesion.stage === 'Stage 2'
                        ? 'bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-300 dark:border-amber-500/30'
                        : 'bg-cyan-50 dark:bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 border-cyan-300 dark:border-cyan-500/30'
                    }`}>
                      {lesion.stage}
                    </span>
                  </td>
                  <td className="py-3 px-4 text-slate-600 dark:text-slate-400 max-w-xs truncate" title={lesion.clinicalRecommendation}>
                    {lesion.clinicalRecommendation}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
};
