import React, { useState, useRef, useEffect } from 'react';
import { 
  Sparkles, 
  X, 
  Send, 
  Bot, 
  User, 
  ShieldAlert, 
  HeartHandshake, 
  RefreshCw,
  SlidersHorizontal,
  Info,
  Layers,
  HelpCircle
} from 'lucide-react';
import { apiService, detectEmotionalTone } from '../services/api';
import { AssistantMessage, AnalysisResult } from '../types/api';
import { useAIChat, ExplanationMode } from '../context/AIChatContext';

interface AIAssistantDrawerProps {
  isOpen?: boolean;
  onClose?: () => void;
  context?: AnalysisResult | null;
}

export const AIAssistantDrawer: React.FC<AIAssistantDrawerProps> = ({
  isOpen: propIsOpen,
  onClose: propOnClose,
  context: propContext,
}) => {
  const {
    isDrawerOpen: ctxIsOpen,
    closeAssistant: ctxClose,
    activeAnalysis: ctxAnalysis,
    explanationMode,
    setExplanationMode,
    sessionId,
    resetSession,
    pendingPrompt,
    clearPendingPrompt,
  } = useAIChat();

  const isOpen = propIsOpen !== undefined ? propIsOpen : ctxIsOpen;
  const onClose = propOnClose || ctxClose;
  const currentCase = propContext !== undefined ? propContext : ctxAnalysis;

  const [messages, setMessages] = useState<AssistantMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: 'Hello. I am your Gemini-powered Dental AI Assistant. I can explain radiographic findings, MLUA segmentation masks, validation metrics, and model limitations. How can I assist you today?',
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
    },
  ]);
  const [inputValue, setInputValue] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  // Handle pending prompt passed from other UI components
  useEffect(() => {
    if (pendingPrompt && isOpen) {
      setInputValue(pendingPrompt);
      clearPendingPrompt();
    }
  }, [pendingPrompt, isOpen, clearPendingPrompt]);

  // Reset conversation when session changes (e.g., when a new case is selected)
  useEffect(() => {
    setMessages([
      {
        id: `welcome-${Date.now()}`,
        sender: 'assistant',
        text: currentCase?.findings && currentCase.findings.length > 0
          ? `I am ready to explain the findings from the current analyzed case. The MLUA model identified ${currentCase.findings.length} suspected caries candidate(s). What would you like me to explain?`
          : 'Hello. I am your Dental AI Assistant. You can ask me general questions about the MLUA model, validation metrics, or upload an X-ray to interpret specific findings.',
        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      },
    ]);
  }, [sessionId]);

  const handleSend = async (textToSend?: string) => {
    const text = (textToSend || inputValue).trim();
    if (!text || isLoading) return;

    const userEmotion = detectEmotionalTone(text);
    const userMsg: AssistantMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      detectedEmotion: userEmotion,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const reply = await apiService.sendAssistantMessage(
        text,
        currentCase,
        sessionId,
        explanationMode
      );
      setMessages((prev) => [...prev, reply]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          sender: 'assistant',
          text: 'I apologize, but I am temporarily unable to connect to the medical assistant service. Please consult your supervising clinician or dental practitioner directly.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearHistory = () => {
    resetSession();
  };

  // Context-aware quick suggested questions
  const getSuggestedQuestions = () => {
    if (currentCase?.findings && currentCase.findings.length > 0) {
      return [
        "Explain this X-ray result",
        "What does the highlighted region mean?",
        `What does ${currentCase.stage.level || 'this staging'} indicate?`,
        "How does the MLUA model detect caries?",
        "What does the Dice score mean?",
        "Why can the model produce false positives?",
        "What are the limitations of this prediction?",
      ];
    }
    return [
      "How does the MLUA model detect caries?",
      "What does the Dice score mean?",
      "Why can the model produce false positives?",
      "What are the limitations of this prediction?",
      "What is cervical burnout?",
      "How do patch-based predictions work?",
    ];
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
      {/* Backdrop */}
      <div 
        onClick={onClose}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
      />

      {/* Drawer Body */}
      <div className="relative w-full max-w-md sm:max-w-lg bg-white dark:bg-[#0d1322] border-l border-slate-200 dark:border-[#1b2742] shadow-2xl flex flex-col h-full z-10 animate-slide-left transition-colors">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 dark:border-[#1b2742] flex items-center justify-between bg-slate-50 dark:bg-[#070a12]">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-gradient-to-tr from-cyan-600 to-cyan-400 text-slate-950 font-bold shadow-md shadow-cyan-500/20">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-slate-900 dark:text-slate-100 text-sm flex items-center gap-1.5">
                Dental AI Assistant
                <span className="text-[10px] px-1.5 py-0.5 rounded-full bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 border border-cyan-500/20 font-medium">
                  Gemini &bull; MLUA
                </span>
              </h3>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                Decision Support &bull; Context-Aware Explanation
              </p>
            </div>
          </div>

          <div className="flex items-center gap-1">
            <button
              onClick={handleClearHistory}
              title="Reset Conversation"
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-800 transition"
              aria-label="Reset conversation"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
            <button
              onClick={onClose}
              className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-800 transition"
              aria-label="Close assistant drawer"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Case & Mode Bar */}
        <div className="px-4 py-2 bg-slate-100 dark:bg-[#0a0f1d] border-b border-slate-200 dark:border-[#1b2742] flex items-center justify-between text-xs">
          <div className="flex items-center gap-1.5 text-slate-600 dark:text-slate-300 font-medium truncate max-w-[200px]">
            <Layers className="w-3.5 h-3.5 text-cyan-500 shrink-0" />
            <span className="truncate">
              {currentCase?.id ? `Case: ${currentCase.patientPseudoId || currentCase.id}` : 'No Case Loaded (General Mode)'}
            </span>
          </div>

          <div className="flex items-center gap-1 bg-white dark:bg-[#121b2d] p-0.5 rounded-lg border border-slate-200 dark:border-[#1b2742]">
            {(['simple', 'standard', 'technical'] as ExplanationMode[]).map((m) => (
              <button
                key={m}
                onClick={() => setExplanationMode(m)}
                className={`px-2 py-0.5 rounded text-[10px] font-semibold capitalize transition ${
                  explanationMode === m
                    ? 'bg-cyan-600 text-white shadow-xs'
                    : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {m}
              </button>
            ))}
          </div>
        </div>

        {/* Clinical Governance Disclaimer Banner */}
        <div className="px-4 py-2 bg-amber-50 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-900/40 text-[11px] text-amber-800 dark:text-amber-300 flex items-start gap-2">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <span>
            Decision-support explanations only. Does not replace autonomous diagnosis. MLUA segmentation is the ground-truth for candidate localization.
          </span>
        </div>

        {/* Message Stream */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.map((msg) => (
            <div
              key={msg.id}
              className={`flex gap-3 ${msg.sender === 'user' ? 'justify-end' : 'justify-start'}`}
            >
              {msg.sender === 'assistant' && (
                <div className="w-7 h-7 rounded-lg bg-gradient-to-tr from-cyan-600 to-cyan-400 text-slate-950 flex items-center justify-center font-bold shrink-0 mt-0.5 shadow-sm">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div className="max-w-[85%] space-y-1">
                <div
                  className={`p-3.5 rounded-2xl text-xs leading-relaxed whitespace-pre-wrap ${
                    msg.sender === 'user'
                      ? 'bg-cyan-600 text-white rounded-br-none shadow-sm shadow-cyan-600/20'
                      : 'bg-slate-100 dark:bg-[#121b2d] border border-slate-200 dark:border-[#1b2742] text-slate-800 dark:text-slate-200 rounded-bl-none'
                  }`}
                >
                  {msg.text}
                </div>

                {/* Emotional Feedback Badge */}
                {msg.detectedEmotion && msg.detectedEmotion !== 'neutral' && msg.sender === 'user' && (
                  <div className="flex items-center gap-1 text-[10px] text-slate-400 justify-end">
                    <HeartHandshake className="w-3 h-3 text-cyan-500" />
                    <span className="capitalize">{msg.detectedEmotion} tone detected — empathetic mode active</span>
                  </div>
                )}

                <span className={`block text-[10px] text-slate-400 ${msg.sender === 'user' ? 'text-right' : 'text-left'}`}>
                  {msg.timestamp}
                </span>
              </div>

              {msg.sender === 'user' && (
                <div className="w-7 h-7 rounded-lg bg-cyan-600 flex items-center justify-center text-white shrink-0 mt-0.5 shadow-sm">
                  <User className="w-4 h-4" />
                </div>
              )}
            </div>
          ))}

          {isLoading && (
            <div className="flex gap-3 items-center text-xs text-slate-400">
              <div className="w-7 h-7 rounded-lg bg-cyan-100 dark:bg-cyan-900/40 flex items-center justify-center text-cyan-500 animate-pulse">
                <Bot className="w-4 h-4" />
              </div>
              <span className="animate-pulse">Consulting Gemini & MLUA Case Knowledge...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Suggested Questions */}
        <div className="p-3 border-t border-slate-200 dark:border-[#1b2742] bg-slate-50 dark:bg-[#070a12]">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1.5">
            Suggested Clinical Questions
          </span>
          <div className="flex flex-wrap gap-1.5 max-h-24 overflow-y-auto">
            {getSuggestedQuestions().map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => handleSend(prompt)}
                disabled={isLoading}
                className="text-[11px] px-2.5 py-1 rounded-lg bg-white dark:bg-[#121b2d] border border-slate-200 dark:border-[#1b2742] text-slate-700 dark:text-slate-300 hover:border-cyan-500 dark:hover:border-cyan-500 hover:text-cyan-600 dark:hover:text-cyan-400 transition text-left cursor-pointer"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>

        {/* Input Bar */}
        <form 
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }} 
          className="p-3 border-t border-slate-200 dark:border-[#1b2742] bg-white dark:bg-[#0d1322] flex gap-2"
        >
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask about this result, MLUA architecture, or validation metrics..."
            className="flex-1 px-3.5 py-2.5 bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-xl text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:border-cyan-500"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-xl text-xs font-semibold flex items-center justify-center transition cursor-pointer"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
};
