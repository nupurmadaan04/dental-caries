import React, { useState, useRef, useEffect } from 'react';
import { 
  Sparkles, 
  X, 
  Send, 
  Bot, 
  User, 
  ShieldAlert, 
  HeartHandshake, 
  AlertCircle,
  HelpCircle,
  Stethoscope
} from 'lucide-react';
import { apiService, detectEmotionalTone } from '../services/api';
import { AssistantMessage } from '../types/api';

interface AIAssistantDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  context?: any;
}

export const AIAssistantDrawer: React.FC<AIAssistantDrawerProps> = ({
  isOpen,
  onClose,
  context,
}) => {
  const [messages, setMessages] = useState<AssistantMessage[]>([
    {
      id: 'welcome',
      sender: 'assistant',
      text: 'Hello. I am your Dental AI Clinical Assistant. I can help explain radiographic findings, caries staging criteria, panoramic imaging considerations, and clinical verification workflows. How can I assist you today?',
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

  const handleSend = async (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputValue.trim() || isLoading) return;

    const userText = inputValue.trim();
    const userEmotion = detectEmotionalTone(userText);
    const userMsg: AssistantMessage = {
      id: `user-${Date.now()}`,
      sender: 'user',
      text: userText,
      timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      detectedEmotion: userEmotion,
    };

    setMessages((prev) => [...prev, userMsg]);
    setInputValue('');
    setIsLoading(true);

    try {
      const reply = await apiService.sendAssistantMessage(userText, context);
      setMessages((prev) => [...prev, reply]);
    } catch {
      setMessages((prev) => [
        ...prev,
        {
          id: `err-${Date.now()}`,
          sender: 'assistant',
          text: 'I apologize, I am temporarily unable to connect to the medical assistant service. Please consult your supervising clinician or dental practitioner directly.',
          timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const quickPrompts = [
    "What does Stage 1 caries mean?",
    "Why does cervical burnout look like caries?",
    "Can panoramic scans miss early enamel decay?",
    "How does the doctor verification form work?",
  ];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 overflow-hidden flex justify-end">
      {/* Backdrop */}
      <div 
        onClick={onClose}
        className="fixed inset-0 bg-black/60 backdrop-blur-sm transition-opacity"
      />

      {/* Drawer Body */}
      <div className="relative w-full max-w-md bg-white dark:bg-[#0d1322] border-l border-slate-200 dark:border-[#1b2742] shadow-2xl flex flex-col h-full z-10 animate-slide-left transition-colors">
        {/* Header */}
        <div className="p-4 border-b border-slate-200 dark:border-[#1b2742] flex items-center justify-between bg-slate-50 dark:bg-[#070a12]">
          <div className="flex items-center gap-2.5">
            <div className="p-2 rounded-xl bg-cyan-100 dark:bg-cyan-500/20 text-cyan-600 dark:text-cyan-400">
              <Sparkles className="w-4 h-4" />
            </div>
            <div>
              <h3 className="font-bold text-slate-900 dark:text-slate-100 text-sm flex items-center gap-1.5">
                AI Clinical Assistant
              </h3>
              <p className="text-[11px] text-slate-500 dark:text-slate-400">
                Empathetic Medical Decision Support
              </p>
            </div>
          </div>

          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-slate-400 hover:text-slate-700 dark:hover:text-slate-200 hover:bg-slate-200 dark:hover:bg-slate-800 transition"
            aria-label="Close assistant drawer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Clinical Disclaimer Banner */}
        <div className="px-4 py-2 bg-amber-50 dark:bg-amber-950/30 border-b border-amber-200 dark:border-amber-900/40 text-[11px] text-amber-800 dark:text-amber-300 flex items-start gap-2">
          <ShieldAlert className="w-3.5 h-3.5 text-amber-600 dark:text-amber-400 shrink-0 mt-0.5" />
          <span>
            AI-generated information is for educational and decision-support purposes and does not replace professional dental evaluation.
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
                <div className="w-7 h-7 rounded-lg bg-cyan-100 dark:bg-cyan-900/40 border border-cyan-300 dark:border-cyan-700/50 flex items-center justify-center text-cyan-600 dark:text-cyan-400 shrink-0 mt-0.5">
                  <Bot className="w-4 h-4" />
                </div>
              )}

              <div className="max-w-[82%] space-y-1">
                <div
                  className={`p-3.5 rounded-2xl text-xs leading-relaxed ${
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
                <div className="w-7 h-7 rounded-lg bg-cyan-600 flex items-center justify-center text-white shrink-0 mt-0.5">
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
              <span>Consulting clinical knowledge base...</span>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Quick Prompts */}
        <div className="p-3 border-t border-slate-200 dark:border-[#1b2742] bg-slate-50 dark:bg-[#070a12]">
          <span className="text-[10px] font-semibold text-slate-400 uppercase tracking-wider block mb-1.5">
            Suggested Clinical Questions
          </span>
          <div className="flex flex-wrap gap-1.5">
            {quickPrompts.map((prompt, idx) => (
              <button
                key={idx}
                onClick={() => {
                  setInputValue(prompt);
                }}
                className="text-[11px] px-2.5 py-1 rounded-lg bg-white dark:bg-[#121b2d] border border-slate-200 dark:border-[#1b2742] text-slate-700 dark:text-slate-300 hover:border-cyan-500 dark:hover:border-cyan-500 transition text-left"
              >
                {prompt}
              </button>
            ))}
          </div>
        </div>

        {/* Input Bar */}
        <form onSubmit={handleSend} className="p-3 border-t border-slate-200 dark:border-[#1b2742] bg-white dark:bg-[#0d1322] flex gap-2">
          <input
            type="text"
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder="Ask a question about findings, staging, or OPG imaging..."
            className="flex-1 px-3.5 py-2.5 bg-slate-100 dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-xl text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:border-cyan-500"
          />
          <button
            type="submit"
            disabled={!inputValue.trim() || isLoading}
            className="px-4 py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white rounded-xl text-xs font-semibold flex items-center justify-center transition"
          >
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
};
