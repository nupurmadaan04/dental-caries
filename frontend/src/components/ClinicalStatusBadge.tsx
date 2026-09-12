import React from 'react';
import { ShieldCheck, Clock, AlertTriangle, CheckCircle2 } from 'lucide-react';

interface ClinicalStatusBadgeProps {
  status: 'CLINICALLY_VERIFIED' | 'PENDING_REVIEW' | 'PRIORITY_REVIEW' | 'ROUTINE_CHECKUP';
}

export const ClinicalStatusBadge: React.FC<ClinicalStatusBadgeProps> = ({ status }) => {
  switch (status) {
    case 'CLINICALLY_VERIFIED':
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-emerald-50 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border border-emerald-300 dark:border-emerald-500/30 text-xs font-semibold">
          <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 dark:text-emerald-400" />
          <span>Clinically Verified</span>
        </span>
      );
    case 'PRIORITY_REVIEW':
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-rose-50 dark:bg-rose-500/10 text-rose-700 dark:text-rose-400 border border-rose-300 dark:border-rose-500/30 text-xs font-semibold animate-pulse">
          <AlertTriangle className="w-3.5 h-3.5 text-rose-600 dark:text-rose-400" />
          <span>Priority Clinical Follow-up</span>
        </span>
      );
    case 'PENDING_REVIEW':
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-amber-50 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border border-amber-300 dark:border-amber-500/30 text-xs font-semibold">
          <Clock className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400" />
          <span>Pending Clinical Review</span>
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-slate-100 dark:bg-slate-500/10 text-slate-700 dark:text-slate-400 border border-slate-300 dark:border-slate-500/30 text-xs font-semibold">
          <span>Routine Prophylactic Follow-up</span>
        </span>
      );
  }
};
