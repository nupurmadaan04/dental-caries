import React, { useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  ScanLine,
  History,
  FileText,
  Binary,
  ShieldCheck,
  AlertTriangle,
  HelpCircle,
  Settings,
  ChevronLeft,
  ChevronRight,
  ShieldAlert,
  Activity,
  Sparkles,
  X
} from 'lucide-react';
import { PRODUCT_INFO } from '../constants/clinicalMetadata';

interface SidebarProps {
  isCollapsed?: boolean;
  toggleCollapse?: () => void;
  isMobileOpen?: boolean;
  closeMobile?: () => void;
  onOpenAssistant?: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isCollapsed: controlledCollapsed,
  toggleCollapse: controlledToggle,
  isMobileOpen = false,
  closeMobile,
  onOpenAssistant,
}) => {
  const [internalCollapsed, setInternalCollapsed] = useState(false);
  const isControlled = typeof controlledCollapsed === 'boolean';
  const collapsed = isControlled ? controlledCollapsed : internalCollapsed;

  const handleToggle = () => {
    if (isControlled && controlledToggle) {
      controlledToggle();
    } else {
      setInternalCollapsed(!internalCollapsed);
    }
  };

  const navItems = [
    { to: '/', label: 'Dashboard', icon: LayoutDashboard },
    { to: '/analyze', label: 'New Analysis', icon: ScanLine },
    { to: '/history', label: 'History Archive', icon: History },
    { to: '/reports', label: 'Reports & Exports', icon: FileText },
    { to: '/methodology', label: 'Methods', icon: Binary },
    { to: '/verification', label: 'Evaluation & Verification', icon: ShieldCheck },
    { to: '/limitations', label: 'Clinical Limitations', icon: AlertTriangle },
    { to: '/faq', label: 'Clinical FAQ', icon: HelpCircle },
    { to: '/settings', label: 'Settings', icon: Settings },
  ];

  return (
    <>
      {/* Mobile Backdrop */}
      {isMobileOpen && (
        <div
          onClick={closeMobile}
          className="fixed inset-0 z-40 bg-black/70 backdrop-blur-sm lg:hidden transition-opacity"
        />
      )}

      <aside
        className={`fixed lg:static top-0 left-0 z-50 h-screen transition-all duration-300 ease-in-out flex flex-col bg-white dark:bg-[#0d1322] border-r border-slate-200 dark:border-[#1b2742] shrink-0 ${
          isMobileOpen ? 'translate-x-0 w-64' : '-translate-x-full lg:translate-x-0'
        } ${collapsed ? 'lg:w-20' : 'lg:w-64'}`}
      >
        {/* Brand Header */}
        <div className="h-16 flex items-center justify-between px-4 border-b border-slate-200 dark:border-[#1b2742]">
          {!collapsed ? (
            <div className="flex items-center space-x-3 overflow-hidden">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-600 to-cyan-400 flex items-center justify-center shadow-lg shadow-cyan-500/20 text-slate-950 font-bold shrink-0">
                <Activity className="w-5 h-5 text-[#070a12]" />
              </div>
              <div className="truncate">
                <div className="text-sm font-bold text-slate-900 dark:text-slate-100 flex items-center gap-1.5 truncate">
                  {PRODUCT_INFO.shortName}
                </div>
                <div className="text-[11px] text-slate-500 dark:text-slate-400 font-medium truncate">
                  Clinical AI System
                </div>
              </div>
            </div>
          ) : (
            <div className="mx-auto w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-600 to-cyan-400 flex items-center justify-center text-[#070a12] font-bold shrink-0" title={PRODUCT_INFO.name}>
              <Activity className="w-5 h-5 text-[#070a12]" />
            </div>
          )}

          {/* Desktop Collapse Toggle */}
          <button
            onClick={handleToggle}
            className="hidden lg:flex p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-[#121b2d] border border-transparent transition-colors"
            title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
          >
            {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
          </button>

          {/* Mobile Close Button */}
          {closeMobile && (
            <button
              onClick={closeMobile}
              className="lg:hidden p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-white"
            >
              <X className="w-5 h-5" />
            </button>
          )}
        </div>

        {/* Navigation Items */}
        <div className="flex-1 overflow-y-auto py-4 px-3 space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={closeMobile}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-xl text-sm font-medium transition-all group ${
                    isActive
                      ? 'bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/30 shadow-sm shadow-cyan-500/10 font-semibold'
                      : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 hover:bg-slate-100 dark:hover:bg-[#121b2d] border border-transparent'
                  }`
                }
              >
                <Icon className="w-5 h-5 shrink-0 transition-transform group-hover:scale-110" />
                {!collapsed && (
                  <div className="flex items-center justify-between flex-1 truncate">
                    <span>{item.label}</span>
                  </div>
                )}
              </NavLink>
            );
          })}
        </div>

        {/* AI Assistant Button in Left Panel */}
        {onOpenAssistant && (
          <div className="px-3 pb-2">
            <button
              onClick={onOpenAssistant}
              className={`w-full flex items-center gap-2.5 px-3 py-2.5 rounded-xl bg-cyan-50/80 dark:bg-cyan-500/10 border border-cyan-200/80 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-300 hover:bg-cyan-100 dark:hover:bg-cyan-500/20 text-xs font-semibold shadow-sm transition-all ${
                collapsed ? 'justify-center px-0' : ''
              }`}
              title="Open Clinical AI Assistant"
            >
              <Sparkles className="w-4 h-4 text-cyan-600 dark:text-cyan-400 shrink-0" />
              {!collapsed && <span>AI Assistant</span>}
            </button>
          </div>
        )}

        {/* Medical Decision Support Pill */}
        <div className="p-3 border-t border-slate-200 dark:border-[#1b2742]">
          {!collapsed ? (
            <div className="bg-slate-50 dark:bg-[#121b2d] border border-slate-200 dark:border-[#1b2742] rounded-xl p-3">
              <div className="flex items-center gap-1.5 text-xs font-semibold text-cyan-600 dark:text-cyan-400 mb-1">
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Decision Support</span>
              </div>
              <p className="text-[11px] text-slate-500 dark:text-slate-400 leading-snug">
                For dental clinical review & research decision support.
              </p>
            </div>
          ) : (
            <div className="flex justify-center text-cyan-500" title="Clinical Decision Support">
              <ShieldAlert className="w-4 h-4" />
            </div>
          )}
        </div>
      </aside>
    </>
  );
};
