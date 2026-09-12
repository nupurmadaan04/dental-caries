import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  subtitle?: string;
  icon: LucideIcon;
  badge?: string;
  badgeType?: 'brand' | 'success' | 'warning' | 'purple';
  trend?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  subtitle,
  icon: Icon,
  badge,
  badgeType = 'brand',
  trend,
}) => {
  const getBadgeStyle = () => {
    switch (badgeType) {
      case 'success':
        return 'bg-emerald-100 dark:bg-emerald-500/10 text-emerald-700 dark:text-emerald-400 border-emerald-300 dark:border-emerald-500/30';
      case 'warning':
        return 'bg-amber-100 dark:bg-amber-500/10 text-amber-700 dark:text-amber-400 border-amber-300 dark:border-amber-500/30';
      case 'purple':
        return 'bg-purple-100 dark:bg-purple-500/10 text-purple-700 dark:text-purple-400 border-purple-300 dark:border-purple-500/30';
      default:
        return 'bg-cyan-100 dark:bg-cyan-500/10 text-cyan-700 dark:text-cyan-400 border-cyan-300 dark:border-cyan-500/30';
    }
  };

  return (
    <div className="bg-white dark:bg-[#121b2d]/80 backdrop-blur-md border border-slate-200 dark:border-[#1b2742] hover:border-cyan-500/40 dark:hover:border-cyan-500/40 rounded-2xl p-5 shadow-md dark:shadow-lg transition-all duration-200 hover:-translate-y-0.5 group">
      <div className="flex items-start justify-between mb-3">
        <div className="w-10 h-10 rounded-xl bg-slate-100 dark:bg-[#1a243c] border border-slate-200 dark:border-[#233354] flex items-center justify-center text-cyan-600 dark:text-cyan-400 group-hover:scale-110 transition-transform">
          <Icon className="w-5 h-5" />
        </div>
        {badge && (
          <span className={`text-[10px] font-mono px-2 py-0.5 rounded-full border ${getBadgeStyle()}`}>
            {badge}
          </span>
        )}
      </div>

      <div className="space-y-1">
        <div className="text-xs font-semibold text-slate-500 dark:text-slate-400 uppercase tracking-wider">{title}</div>
        <div className="text-2xl font-bold text-slate-900 dark:text-slate-100 font-mono flex items-baseline gap-2">
          <span>{value}</span>
        </div>
        {subtitle && <div className="text-xs text-slate-600 dark:text-slate-400">{subtitle}</div>}
        {trend && <div className="text-[11px] text-cyan-600 dark:text-cyan-300 font-mono mt-1">{trend}</div>}
      </div>
    </div>
  );
};
