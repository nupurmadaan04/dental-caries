import React, { useState } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { Sidebar } from './components/Sidebar';
import { Navbar } from './components/Navbar';
import { AIAssistantDrawer } from './components/AIAssistantDrawer';

// Pages
import { DashboardPage } from './pages/DashboardPage';
import { NewAnalysisPage } from './pages/NewAnalysisPage';
import { AnalysisResultPage } from './pages/AnalysisResultPage';
import { HistoryPage } from './pages/HistoryPage';
import { ReportsPage } from './pages/ReportsPage';
import { MLUAMethodologyPage } from './pages/MLUAMethodologyPage';
import { TechnicalVerificationPage } from './pages/TechnicalVerificationPage';
import { LimitationsPage } from './pages/LimitationsPage';
import { FAQPage } from './pages/FAQPage';
import { SettingsPage } from './pages/SettingsPage';
import { PRODUCT_INFO } from './constants/clinicalMetadata';

export const App: React.FC = () => {
  const [isSidebarCollapsed, setIsSidebarCollapsed] = useState<boolean>(false);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState<boolean>(false);
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);

  return (
    <BrowserRouter>
      <div className="min-h-screen bg-slate-50 dark:bg-[#070a12] text-slate-900 dark:text-slate-100 flex flex-col font-sans antialiased selection:bg-cyan-500/30 selection:text-cyan-600 dark:selection:text-cyan-200 transition-colors">
        <div className="flex flex-1 relative overflow-hidden">
          {/* Sidebar */}
          <Sidebar
            isCollapsed={isSidebarCollapsed}
            toggleCollapse={() => setIsSidebarCollapsed(!isSidebarCollapsed)}
            isMobileOpen={isMobileMenuOpen}
            closeMobile={() => setIsMobileMenuOpen(false)}
            onOpenAssistant={() => setIsDrawerOpen(true)}
          />

          {/* Main Layout Area */}
          <div className="flex-1 flex flex-col min-w-0 h-screen overflow-y-auto">
            <Navbar
              onOpenMobileMenu={() => setIsMobileMenuOpen(true)}
            />

            {/* Content Container */}
            <main className="flex-1 p-4 sm:p-6 lg:p-8 max-w-7xl w-full mx-auto">
              <Routes>
                <Route path="/" element={<DashboardPage />} />
                <Route path="/analyze" element={<NewAnalysisPage />} />
                <Route path="/results/:id" element={<AnalysisResultPage />} />
                <Route path="/history" element={<HistoryPage />} />
                <Route path="/reports" element={<ReportsPage />} />
                <Route path="/methodology" element={<MLUAMethodologyPage />} />
                <Route path="/verification" element={<TechnicalVerificationPage />} />
                <Route path="/limitations" element={<LimitationsPage />} />
                <Route path="/faq" element={<FAQPage />} />
                <Route path="/settings" element={<SettingsPage />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Routes>
            </main>

            {/* Clinical Governance Footer */}
            <footer className="border-t border-slate-200 dark:border-[#1b2742] py-4 px-6 text-center text-xs text-slate-500 dark:text-slate-400 flex flex-col sm:flex-row items-center justify-between gap-2 bg-white dark:bg-[#070a12] transition-colors">
              <div className="flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
                <span>{PRODUCT_INFO.name} v{PRODUCT_INFO.version} &bull; {PRODUCT_INFO.modality}</span>
              </div>
              <p className="text-[11px] text-slate-400 dark:text-slate-500">
                Clinical Decision Support &bull; For Dental Practitioner Review Only
              </p>
            </footer>
          </div>
        </div>

        {/* AI Clinical Assistant Drawer */}
        <AIAssistantDrawer isOpen={isDrawerOpen} onClose={() => setIsDrawerOpen(false)} />
      </div>
    </BrowserRouter>
  );
};
