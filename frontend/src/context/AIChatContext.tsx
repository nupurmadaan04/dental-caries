import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { AnalysisResult } from '../types/api';

export type ExplanationMode = 'standard' | 'simple' | 'technical';

interface AIChatContextType {
  isDrawerOpen: boolean;
  openAssistant: (initialPrompt?: string) => void;
  closeAssistant: () => void;
  activeAnalysis: AnalysisResult | null;
  setActiveAnalysis: (analysis: AnalysisResult | null) => void;
  explanationMode: ExplanationMode;
  setExplanationMode: (mode: ExplanationMode) => void;
  sessionId: string;
  resetSession: () => void;
  pendingPrompt: string | null;
  clearPendingPrompt: () => void;
}

const AIChatContext = createContext<AIChatContextType | undefined>(undefined);

export const AIChatProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [isDrawerOpen, setIsDrawerOpen] = useState<boolean>(false);
  const [activeAnalysis, setActiveAnalysisState] = useState<AnalysisResult | null>(null);
  const [explanationMode, setExplanationMode] = useState<ExplanationMode>('standard');
  const [sessionId, setSessionId] = useState<string>(() => `session-${Date.now()}`);
  const [pendingPrompt, setPendingPrompt] = useState<string | null>(null);

  const setActiveAnalysis = useCallback((analysis: AnalysisResult | null) => {
    setActiveAnalysisState(analysis);
    if (analysis) {
      // Set dedicated session ID for the active case to ensure strict case isolation
      setSessionId(`session-case-${analysis.id}`);
    } else {
      setSessionId(`session-general-${Date.now()}`);
    }
  }, []);

  const openAssistant = useCallback((initialPrompt?: string) => {
    if (initialPrompt) {
      setPendingPrompt(initialPrompt);
    }
    setIsDrawerOpen(true);
  }, []);

  const closeAssistant = useCallback(() => {
    setIsDrawerOpen(false);
  }, []);

  const resetSession = useCallback(() => {
    const newId = activeAnalysis ? `session-case-${activeAnalysis.id}-${Date.now()}` : `session-${Date.now()}`;
    setSessionId(newId);
  }, [activeAnalysis]);

  const clearPendingPrompt = useCallback(() => {
    setPendingPrompt(null);
  }, []);

  return (
    <AIChatContext.Provider
      value={{
        isDrawerOpen,
        openAssistant,
        closeAssistant,
        activeAnalysis,
        setActiveAnalysis,
        explanationMode,
        setExplanationMode,
        sessionId,
        resetSession,
        pendingPrompt,
        clearPendingPrompt,
      }}
    >
      {children}
    </AIChatContext.Provider>
  );
};

export const useAIChat = () => {
  const ctx = useContext(AIChatContext);
  if (!ctx) throw new Error('useAIChat must be used within AIChatProvider');
  return ctx;
};
