import React from 'react';
import { CariesDepth } from '../types/api';

export const StagingBadge: React.FC<{ depth: CariesDepth; size?: 'sm' | 'md' }> = ({ depth, size = 'sm' }) => {
  const getStyle = () => {
    switch (depth) {
      case 'Deep Dentin / Pulp (D3)':
        return 'bg-red-500/15 text-red-400 border-red-500/30';
      case 'Dentin (D1/D2)':
        return 'bg-amber-500/15 text-amber-400 border-amber-500/30';
      case 'Enamel (E1/E2)':
        return 'bg-cyan-500/15 text-cyan-400 border-cyan-500/30';
      default:
        return 'bg-slate-500/15 text-slate-400 border-slate-500/30';
    }
  };

  const pad = size === 'sm' ? 'px-2 py-0.5 text-[11px]' : 'px-3 py-1 text-xs font-semibold';

  return (
    <span className={`inline-flex items-center gap-1 rounded-full font-mono border ${getStyle()} ${pad}`}>
      <span className="w-1.5 h-1.5 rounded-full bg-current opacity-80 animate-pulse" />
      <span>{depth}</span>
    </span>
  );
};
