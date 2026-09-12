import React, { useState, useEffect } from 'react';
import {
  Settings,
  Server,
  Sun,
  Moon,
  CheckCircle2,
  AlertCircle,
  Sliders,
  ShieldCheck,
  Activity,
  RotateCcw,
  Stethoscope
} from 'lucide-react';
import { useTheme } from '../context/ThemeContext';
import { apiService } from '../services/api';
import { HealthStatus } from '../types/api';
import { PRODUCT_INFO } from '../constants/clinicalMetadata';

export const SettingsPage: React.FC = () => {
  const { theme, toggleTheme, setTheme } = useTheme();

  const [defaultThreshold, setDefaultThreshold] = useState<number>(
    parseFloat(localStorage.getItem('dental_caries_threshold') || '0.50')
  );
  const [health, setHealth] = useState<HealthStatus>({
    status: 'ONLINE',
    isLiveBackend: true,
    serviceName: 'Dental Caries Clinical AI Service',
    version: '2.4.0',
    device: 'Local Station',
  });
  const [saveSuccess, setSaveSuccess] = useState(false);

  useEffect(() => {
    apiService.getHealth().then(setHealth).catch(() => {});
  }, []);

  const handleSavePreferences = (e: React.FormEvent) => {
    e.preventDefault();
    localStorage.setItem('dental_caries_threshold', defaultThreshold.toString());
    setSaveSuccess(true);
    setTimeout(() => setSaveSuccess(false), 3000);
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto pb-12">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 text-xs font-semibold mb-2">
          <Settings className="w-3.5 h-3.5" />
          <span>System & Interface Configuration</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
          Settings
        </h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Configure clinical screening preferences, visual theme, and view analysis service health.
        </p>
      </div>

      <form onSubmit={handleSavePreferences} className="space-y-6">
        {/* Read-Only Analysis Service Status (NOT EDITABLE) */}
        <div className="p-6 rounded-3xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-200 dark:border-[#1b2742] pb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400">
                <Server className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                  Analysis Service Status
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Read-only production clinical inference service status
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2">
              <span
                className={`w-2.5 h-2.5 rounded-full ${
                  health.status === 'ONLINE' ? 'bg-emerald-500 animate-pulse' : 'bg-amber-500'
                }`}
              />
              <span className="text-xs font-bold text-slate-800 dark:text-slate-200">
                {health.status === 'ONLINE' ? 'ONLINE / CONNECTED' : 'READY / STANDALONE'}
              </span>
            </div>
          </div>

          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs">
            <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
              <span className="text-slate-400 text-[10px] block font-semibold uppercase">SERVICE NAME</span>
              <span className="font-semibold text-slate-800 dark:text-slate-200">{health.serviceName}</span>
            </div>
            <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
              <span className="text-slate-400 text-[10px] block font-semibold uppercase">VERSION</span>
              <span className="font-mono text-slate-800 dark:text-slate-200 font-semibold">{PRODUCT_INFO.version}</span>
            </div>
            <div className="p-3.5 rounded-2xl bg-slate-50 dark:bg-slate-950 border border-slate-200 dark:border-slate-800">
              <span className="text-slate-400 text-[10px] block font-semibold uppercase">MODALITY</span>
              <span className="font-semibold text-slate-800 dark:text-slate-200">{PRODUCT_INFO.modality}</span>
            </div>
          </div>
        </div>

        {/* Visual Theme Selection (Dark / Light) */}
        <div className="p-6 rounded-3xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-200 dark:border-[#1b2742] pb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-amber-50 dark:bg-cyan-500/10 text-amber-600 dark:text-cyan-400">
                {theme === 'dark' ? <Moon className="w-5 h-5" /> : <Sun className="w-5 h-5" />}
              </div>
              <div>
                <h3 className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                  Interface Visual Theme
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Switch between high-contrast darkroom mode and clinic daytime light mode
                </p>
              </div>
            </div>

            <div className="flex bg-slate-100 dark:bg-slate-950 p-1 rounded-xl border border-slate-200 dark:border-slate-800">
              <button
                type="button"
                onClick={() => setTheme('light')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  theme === 'light'
                    ? 'bg-white text-slate-900 shadow-sm border border-slate-200 font-bold'
                    : 'text-slate-500 hover:text-slate-900'
                }`}
              >
                <Sun className="w-3.5 h-3.5 text-amber-500" />
                <span>Light</span>
              </button>
              <button
                type="button"
                onClick={() => setTheme('dark')}
                className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-xs font-semibold transition ${
                  theme === 'dark'
                    ? 'bg-[#121b2d] text-cyan-400 shadow-sm border border-[#1b2742] font-bold'
                    : 'text-slate-400 hover:text-slate-100'
                }`}
              >
                <Moon className="w-3.5 h-3.5 text-cyan-400" />
                <span>Dark</span>
              </button>
            </div>
          </div>
          <p className="text-xs text-slate-500 dark:text-slate-400">
            Theme selection is immediately applied to all cards, tables, viewers, modals, and persisted in browser storage.
          </p>
        </div>

        {/* Default Caries Threshold Preference */}
        <div className="p-6 rounded-3xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] space-y-4 shadow-sm">
          <div className="flex items-center justify-between border-b border-slate-200 dark:border-[#1b2742] pb-4">
            <div className="flex items-center gap-3">
              <div className="p-2 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400">
                <Sliders className="w-5 h-5" />
              </div>
              <div>
                <h3 className="font-bold text-slate-900 dark:text-slate-100 text-sm">
                  Default Detection Sensitivity
                </h3>
                <p className="text-xs text-slate-500 dark:text-slate-400">
                  Preset detection threshold used when opening new radiograph analyses
                </p>
              </div>
            </div>

            <span className="font-mono text-sm font-bold text-cyan-600 dark:text-cyan-400 bg-cyan-50 dark:bg-cyan-500/10 px-2.5 py-1 rounded-full border border-cyan-200 dark:border-cyan-500/30">
              {defaultThreshold.toFixed(2)}
            </span>
          </div>

          <div className="space-y-3">
            <input
              type="range"
              min="0.10"
              max="0.90"
              step="0.05"
              value={defaultThreshold}
              onChange={(e) => setDefaultThreshold(parseFloat(e.target.value))}
              className="w-full accent-cyan-600 h-2 bg-slate-200 dark:bg-slate-800 rounded-lg cursor-pointer"
            />
            <div className="flex justify-between text-[11px] text-slate-400">
              <span>0.10 (High Sensitivity)</span>
              <span>0.50 (Standard Preset)</span>
              <span>0.90 (High Specificity)</span>
            </div>
          </div>
        </div>

        {/* Save Button */}
        <div className="flex items-center justify-between pt-2">
          <div>
            {saveSuccess && (
              <span className="inline-flex items-center gap-1.5 text-xs text-emerald-600 dark:text-emerald-400 font-semibold animate-fade-in">
                <CheckCircle2 className="w-4 h-4" />
                Preferences saved successfully.
              </span>
            )}
          </div>

          <button
            type="submit"
            className="px-6 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs shadow-md shadow-cyan-600/20 transition"
          >
            Save Preferences
          </button>
        </div>
      </form>
    </div>
  );
};
