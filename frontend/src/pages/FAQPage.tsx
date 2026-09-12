import React, { useState } from 'react';
import { 
  HelpCircle, 
  ChevronDown, 
  ChevronUp, 
  Search, 
  Stethoscope, 
  ShieldAlert, 
  Sparkles, 
  FileText 
} from 'lucide-react';
import { CLINICAL_FAQS, PRODUCT_INFO } from '../constants/clinicalMetadata';

const FAQ_CATEGORIES = [
  { id: 'all', label: 'All Questions' },
  { id: 'general', label: 'General & Pathology', matchIds: ['q1', 'q2', 'q3', 'q16', 'q17'] },
  { id: 'stages', label: 'Caries Staging', matchIds: ['q4', 'q5', 'q6', 'q7'] },
  { id: 'imaging', label: 'OPG Quality & Artifacts', matchIds: ['q8', 'q9', 'q13', 'q14', 'q15', 'q19'] },
  { id: 'workflow', label: 'Clinical Workflow & Review', matchIds: ['q10', 'q11', 'q12', 'q18', 'q20'] },
];

export const FAQPage: React.FC = () => {
  const [openItems, setOpenItems] = useState<Record<string, boolean>>({
    q1: true,
    q2: true,
    q3: true,
  });
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedCategory, setSelectedCategory] = useState('all');

  const toggleItem = (id: string) => {
    setOpenItems((prev) => ({ ...prev, [id]: !prev[id] }));
  };

  const filteredFaqs = CLINICAL_FAQS.filter((item) => {
    const matchesSearch =
      item.question.toLowerCase().includes(searchQuery.toLowerCase()) ||
      item.answer.toLowerCase().includes(searchQuery.toLowerCase());

    const activeCat = FAQ_CATEGORIES.find((c) => c.id === selectedCategory);
    const matchesCategory =
      selectedCategory === 'all' ||
      (activeCat?.matchIds && activeCat.matchIds.includes(item.id));

    return matchesSearch && matchesCategory;
  });

  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 text-xs font-semibold mb-2">
          <HelpCircle className="w-3.5 h-3.5" />
          <span>Clinical & Patient Knowledge Base</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
          Frequently Asked Questions
        </h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Essential clinical answers regarding dental caries pathology, panoramic radiograph screening, and AI decision support.
        </p>
      </div>

      {/* Category Buttons & Search */}
      <div className="space-y-4">
        {/* Category Pills */}
        <div className="flex flex-wrap items-center gap-2">
          {FAQ_CATEGORIES.map((cat) => {
            const isSelected = selectedCategory === cat.id;
            const count =
              cat.id === 'all'
                ? CLINICAL_FAQS.length
                : CLINICAL_FAQS.filter((f) => cat.matchIds?.includes(f.id)).length;

            return (
              <button
                key={cat.id}
                onClick={() => setSelectedCategory(cat.id)}
                className={`px-3.5 py-1.5 rounded-xl text-xs font-semibold transition flex items-center gap-1.5 ${
                  isSelected
                    ? 'bg-cyan-600 text-white shadow-sm shadow-cyan-600/20'
                    : 'bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] text-slate-600 dark:text-slate-300 hover:border-cyan-500/50 hover:text-cyan-600 dark:hover:text-cyan-400'
                }`}
              >
                <span>{cat.label}</span>
                <span
                  className={`text-[10px] px-1.5 py-0.2 rounded-full font-mono ${
                    isSelected
                      ? 'bg-white/20 text-white'
                      : 'bg-slate-100 dark:bg-slate-800 text-slate-500 dark:text-slate-400'
                  }`}
                >
                  {count}
                </span>
              </button>
            );
          })}
        </div>

        {/* Search Input */}
        <div className="relative max-w-xl">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search clinical questions (e.g. 'Stage 1', 'false positives', 'bitewing', 'cavity')..."
            className="w-full pl-10 pr-4 py-3 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-2xl text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:border-cyan-500 shadow-sm"
          />
        </div>
      </div>

      {/* FAQ Accordion List */}
      <div className="space-y-3">
        {filteredFaqs.map((faq) => {
          const isOpen = !!openItems[faq.id];
          return (
            <div
              key={faq.id}
              className={`rounded-2xl border transition-all ${
                isOpen
                  ? 'bg-white dark:bg-[#0d1322] border-cyan-500/40 shadow-sm'
                  : 'bg-white dark:bg-[#0d1322]/60 border-slate-200 dark:border-[#1b2742] hover:border-slate-300 dark:hover:border-slate-700'
              }`}
            >
              <button
                onClick={() => toggleItem(faq.id)}
                className="w-full flex items-center justify-between p-5 text-left transition"
              >
                <div className="flex items-center gap-3">
                  <span className="w-2 h-2 rounded-full bg-cyan-500 shrink-0" />
                  <span className="font-semibold text-slate-900 dark:text-slate-100 text-sm">
                    {faq.question}
                  </span>
                </div>
                {isOpen ? (
                  <ChevronUp className="w-5 h-5 text-cyan-600 dark:text-cyan-400 shrink-0" />
                ) : (
                  <ChevronDown className="w-5 h-5 text-slate-400 shrink-0" />
                )}
              </button>

              {isOpen && (
                <div className="px-5 pb-5 pt-1 border-t border-slate-100 dark:border-slate-800/60">
                  <p className="text-xs text-slate-600 dark:text-slate-300 leading-relaxed">
                    {faq.answer}
                  </p>
                </div>
              )}
            </div>
          );
        })}

        {filteredFaqs.length === 0 && (
          <div className="p-8 text-center bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-2xl text-xs text-slate-500">
            No clinical FAQ matches found for &quot;{searchQuery}&quot;. Please try a different search term.
          </div>
        )}
      </div>
    </div>
  );
};
