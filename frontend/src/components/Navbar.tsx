import React from 'react';
import { useLocation } from 'react-router-dom';
import {
  Sun,
  Moon,
  Menu
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';

interface NavbarProps {
  onOpenMobileMenu?: () => void;
}

export const Navbar: React.FC<NavbarProps> = ({ 
  onOpenMobileMenu, 
}) => {
  const { theme, toggleTheme } = useTheme();
  const location = useLocation();

  const getPageTitle = (path: string) => {
    if (path === '/') return 'Clinical Dashboard';
    if (path.startsWith('/analyze')) return 'New Panoramic Analysis';
    if (path.startsWith('/results')) return 'Clinical Review & Findings';
    if (path.startsWith('/history')) return 'Radiograph History Archive';
    if (path.startsWith('/reports')) return 'Reports & Exports';
    if (path.startsWith('/methodology')) return 'Methods';
    if (path.startsWith('/verification')) return 'Evaluation & Verification';
    if (path.startsWith('/limitations')) return 'Clinical Limitations';
    if (path.startsWith('/faq')) return 'Clinical FAQ';
    if (path.startsWith('/settings')) return 'Settings';
    return 'Dental Caries Clinical AI';
  };

  return (
    <header className="sticky top-0 z-30 h-16 bg-white/90 dark:bg-[#070a12]/80 backdrop-blur-md border-b border-slate-200 dark:border-[#1b2742] px-4 sm:px-6 flex items-center justify-between transition-colors">
      {/* Left: Mobile Menu & Title */}
      <div className="flex items-center space-x-3">
        {onOpenMobileMenu && (
          <button
            onClick={onOpenMobileMenu}
            className="lg:hidden p-2 rounded-xl text-slate-500 hover:text-slate-900 dark:text-slate-400 dark:hover:text-white hover:bg-slate-100 dark:hover:bg-slate-800 transition"
            aria-label="Open navigation menu"
          >
            <Menu className="w-5 h-5" />
          </button>
        )}
        <h1 className="text-base font-bold text-slate-800 dark:text-slate-100 truncate">
          {getPageTitle(location.pathname)}
        </h1>
        <span className="text-slate-300 dark:text-slate-700 hidden sm:inline">|</span>
        <span className="text-xs text-slate-500 dark:text-slate-400 font-medium hidden md:inline">
          Panoramic Radiography
        </span>
      </div>

      {/* Right Controls: Clean Theme Toggle */}
      <div className="flex items-center space-x-3">
        <button
          onClick={toggleTheme}
          className="p-2 rounded-xl text-slate-500 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 bg-slate-100 dark:bg-[#121b2d] border border-slate-200 dark:border-[#1b2742] transition-colors"
          title={`Switch to ${theme === 'dark' ? 'Light' : 'Dark'} Theme`}
          aria-label="Toggle visual theme"
        >
          {theme === 'dark' ? (
            <Sun className="w-4 h-4 text-amber-400 hover:rotate-45 transition-transform" />
          ) : (
            <Moon className="w-4 h-4 text-cyan-600 hover:-rotate-12 transition-transform" />
          )}
        </button>
      </div>
    </header>
  );
};
