import React from 'react';
import { Sparkles } from 'lucide-react';
import { useAIChat } from '../context/AIChatContext';

export const FloatingChatButton: React.FC = () => {
  const { openAssistant, activeAnalysis } = useAIChat();

  return (
    <button
      onClick={() => openAssistant()}
      className="fixed bottom-6 right-6 z-40 p-3.5 rounded-2xl bg-gradient-to-tr from-cyan-600 to-cyan-400 text-slate-950 font-bold shadow-xl shadow-cyan-500/25 hover:shadow-cyan-500/40 hover:scale-105 active:scale-95 transition-all duration-200 flex items-center gap-2.5 group cursor-pointer"
      title="Open Dental AI Assistant"
      aria-label="Open Dental AI Assistant"
    >
      <Sparkles className="w-5 h-5 text-slate-950 group-hover:rotate-12 transition-transform duration-300" />
      <span className="text-xs font-bold tracking-tight hidden sm:inline text-slate-950">
        {activeAnalysis ? 'Ask AI About Case' : 'AI Assistant'}
      </span>
    </button>
  );
};
