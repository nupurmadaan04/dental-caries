import React from 'react';
import { 
  AlertTriangle, 
  ShieldAlert, 
  Stethoscope, 
  Info,
  CheckCircle2,
  FileWarning
} from 'lucide-react';
import { CLINICAL_LIMITATIONS, PRODUCT_INFO } from '../constants/clinicalMetadata';

export const LimitationsPage: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-500/10 border border-amber-200 dark:border-amber-500/30 text-amber-700 dark:text-amber-400 text-xs font-semibold mb-2">
          <ShieldAlert className="w-3.5 h-3.5" />
          <span>Clinical Governance & Safety Notice</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
          Clinical Limitations
        </h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Essential radiographic constraints, medical limitations, and clinical review requirements for {PRODUCT_INFO.name}.
        </p>
      </div>

      {/* Prominent Safety Alert Banner */}
      <div className="p-6 rounded-3xl bg-amber-50/80 dark:bg-amber-950/20 border-2 border-amber-300 dark:border-amber-700/50 shadow-sm space-y-3">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-xl bg-amber-200 dark:bg-amber-900/50 text-amber-800 dark:text-amber-300">
            <AlertTriangle className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-sm font-extrabold text-amber-900 dark:text-amber-200 uppercase tracking-wide">
              Mandatory Clinical Practitioner Notice
            </h2>
            <p className="text-xs text-amber-800 dark:text-amber-300/90 font-medium">
              Academic Research Decision-Support System — Not an Autonomous Diagnostic Device
            </p>
          </div>
        </div>

        <p className="text-xs text-slate-700 dark:text-slate-300 leading-relaxed pt-1">
          The system identifies image regions that may be consistent with dental caries. These findings should not be interpreted as a definitive diagnosis. The absence of a highlighted region does not guarantee the absence of disease. Clinical examination and professional judgment remain essential.
        </p>
      </div>

      {/* 10 Clinical & System Limitation Cards */}
      <div className="space-y-4">
        <h3 className="text-base font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <FileWarning className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
          <span>Detailed Clinical Considerations & Imaging Constraints</span>
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 pt-1">
          {CLINICAL_LIMITATIONS.map((item) => (
            <div
              key={item.id}
              className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-2 hover:border-cyan-500/40 transition"
            >
              <h4 className="font-bold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-cyan-500" />
                {item.title}
              </h4>
              <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
                {item.content}
              </p>
            </div>
          ))}
        </div>
      </div>

      {/* Recommended Clinical Protocol */}
      <div className="p-6 rounded-3xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] space-y-3 shadow-sm">
        <h3 className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Stethoscope className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
          <span>Recommended Diagnostic Verification Protocol</span>
        </h3>
        <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
          Whenever candidate caries regions are indicated:
        </p>
        <ul className="text-xs text-slate-700 dark:text-slate-300 space-y-2 list-disc list-inside">
          <li>Perform physical visual-tactile examination under adequate clinical lighting and clean, dry tooth surfaces.</li>
          <li>Assess tooth vitality with thermal or electric pulp testing when dentin involvement is suspected.</li>
          <li>Order supplemental intraoral bitewing or periapical radiographs for high-resolution interproximal confirmation.</li>
          <li>Record clinical findings in the case verification form to maintain a complete diagnostic audit trail.</li>
        </ul>
      </div>
    </div>
  );
};
