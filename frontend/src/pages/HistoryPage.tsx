import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import {
  History,
  Search,
  CheckCircle2,
  Clock,
  ArrowRight,
  Download,
  FileCode,
  FolderOpen
} from 'lucide-react';
import { apiService } from '../services/api';
import { HistoryItem } from '../types/api';

export const HistoryPage: React.FC = () => {
  const [history, setHistory] = useState<HistoryItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<'all' | 'verified' | 'pending'>('all');
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  useEffect(() => {
    apiService.getHistory().then(setHistory);
  }, []);

  const handleDownloadPDF = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDownloadingId(id);
    try {
      await apiService.downloadClinicalReport(id, `dental_caries_clinical_report_${id}.pdf`);
    } catch (err) {
      console.error(err);
    } finally {
      setDownloadingId(null);
    }
  };

  const handleDownloadJSON = async (id: string, e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setDownloadingId(id);
    try {
      await apiService.downloadAnalysisJSON(id, `dental_caries_analysis_${id}.json`);
    } catch (err) {
      console.error(err);
    } finally {
      setDownloadingId(null);
    }
  };

  const filteredHistory = history.filter((item) => {
    const matchesSearch =
      item.id.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.patientId.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter =
      filterStatus === 'all' ||
      (filterStatus === 'verified' && item.isReviewed) ||
      (filterStatus === 'pending' && !item.isReviewed);
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 text-xs font-semibold mb-2">
            <History className="w-3.5 h-3.5" />
            <span>Case History & Radiograph Archive</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
            History Archive
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
            Access past panoramic radiograph screening records, AI findings, and physician verification logs.
          </p>
        </div>
      </div>

      {/* Search & Filter Controls */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3.5 top-3.5" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by Case ID, File, or Patient Ref..."
            className="w-full pl-10 pr-4 py-2.5 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-xl text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:border-cyan-500"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">Filter:</span>
          <div className="flex bg-slate-100 dark:bg-slate-950 p-1 rounded-xl border border-slate-200 dark:border-slate-800">
            {(['all', 'verified', 'pending'] as const).map((st) => (
              <button
                key={st}
                onClick={() => setFilterStatus(st)}
                className={`px-3 py-1 text-xs font-semibold rounded-lg capitalize transition ${
                  filterStatus === st
                    ? 'bg-white dark:bg-[#121b2d] text-cyan-600 dark:text-cyan-400 shadow-sm border border-slate-200 dark:border-[#1b2742]'
                    : 'text-slate-500 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                {st}
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* History Table */}
      <div className="bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-[#121b2d] border-b border-slate-200 dark:border-[#1b2742] text-slate-600 dark:text-slate-400 font-bold uppercase text-[10px]">
                <th className="py-4 px-4 whitespace-nowrap">Case ID</th>
                <th className="py-4 px-4 whitespace-nowrap">Exam Date</th>
                <th className="py-4 px-4">Radiograph File</th>
                <th className="py-4 px-4 whitespace-nowrap text-center">Stage</th>
                <th className="py-4 px-4 whitespace-nowrap">Lesion Count</th>
                <th className="py-4 px-4 whitespace-nowrap">Review Status</th>
                <th className="py-4 px-6 text-right whitespace-nowrap">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-[#1b2742]">
              {filteredHistory.length === 0 ? (
                <tr>
                  <td colSpan={7} className="py-12 text-center text-slate-400 dark:text-slate-500">
                    <FolderOpen className="w-8 h-8 mx-auto mb-2 opacity-50" />
                    <p className="text-sm font-medium">No radiograph records found matching criteria.</p>
                  </td>
                </tr>
              ) : (
                filteredHistory.map((item) => {
                  const isNoCaries = 
                    item.lesionCount === 0 || 
                    item.stageLevel === '0' ||
                    item.stageLevel?.toLowerCase().includes('0') || 
                    item.overallFinding?.toLowerCase().includes('no caries') ||
                    (item as any).finding?.toLowerCase().includes('no caries') ||
                    item.stageLevel?.toLowerCase().includes('no significant');

                  const stageDigit = isNoCaries
                    ? '0'
                    : item.stageLevel?.includes('3') || item.overallFinding?.toLowerCase().includes('extensive')
                    ? '3'
                    : item.stageLevel?.includes('2') || item.overallFinding?.toLowerCase().includes('moderate')
                    ? '2'
                    : '1';

                  const lesionNum = isNoCaries ? 0 : (typeof item.lesionCount === 'number' ? item.lesionCount : ((item as any).lesions ?? 1));

                  const badgeStyle = isNoCaries
                    ? 'bg-emerald-50 dark:bg-emerald-500/10 text-emerald-800 dark:text-emerald-300 border-emerald-200 dark:border-emerald-500/30'
                    : stageDigit === '3'
                    ? 'bg-red-50 dark:bg-red-500/10 text-red-800 dark:text-red-300 border-red-200 dark:border-red-500/30'
                    : stageDigit === '2'
                    ? 'bg-orange-50 dark:bg-orange-500/10 text-orange-800 dark:text-orange-300 border-orange-200 dark:border-orange-500/30'
                    : 'bg-yellow-50 dark:bg-yellow-500/10 text-yellow-800 dark:text-yellow-300 border-yellow-200 dark:border-yellow-500/30';

                  return (
                    <tr key={item.id} className="hover:bg-slate-50 dark:hover:bg-[#121b2d]/50 transition">
                      <td className="py-4 px-4 font-mono font-bold text-cyan-600 dark:text-cyan-400 whitespace-nowrap">
                        {item.id}
                      </td>
                      <td className="py-4 px-4 text-slate-600 dark:text-slate-400 font-mono text-[11px] whitespace-nowrap">
                        {item.timestamp}
                      </td>
                      <td className="py-4 px-4 font-semibold text-slate-900 dark:text-slate-100">
                        <div className="max-w-[200px] truncate" title={item.filename}>
                          {item.filename}
                        </div>
                      </td>
                      <td className="py-4 px-4 whitespace-nowrap text-center">
                        <span className={`inline-flex items-center justify-center px-2.5 py-1 rounded-full text-xs font-bold font-mono border ${badgeStyle}`}>
                          Stage {stageDigit}
                        </span>
                      </td>
                      <td className="py-4 px-4 font-bold font-mono text-slate-900 dark:text-slate-100 whitespace-nowrap">
                        {lesionNum} {lesionNum === 1 ? 'site' : 'sites'} detected
                      </td>
                      <td className="py-4 px-4 whitespace-nowrap">
                        {item.isReviewed ? (
                          <span className="inline-flex items-center gap-1.5 text-xs text-emerald-700 dark:text-emerald-400 font-bold">
                            <CheckCircle2 className="w-4 h-4" />
                            Verified
                          </span>
                        ) : (
                          <Link
                            to={`/results/${item.id}?tab=verification`}
                            className="inline-flex items-center gap-1.5 text-xs text-amber-700 dark:text-amber-400 hover:text-amber-600 dark:hover:text-amber-300 hover:underline font-bold transition group"
                            title="Click to open Physician Review & Sign-Off"
                          >
                            <Clock className="w-4 h-4 group-hover:scale-110 transition-transform" />
                            <span>Pending Review</span>
                          </Link>
                        )}
                      </td>
                      <td className="py-4 px-6 text-right whitespace-nowrap">
                        <div className="inline-flex items-center justify-end gap-2">
                          <button
                            onClick={(e) => handleDownloadPDF(item.id, e)}
                            disabled={downloadingId === item.id}
                            className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:text-cyan-600 dark:hover:text-cyan-400 hover:bg-slate-200 dark:hover:bg-slate-700 transition"
                            title="Download Clinical PDF"
                          >
                            <Download className="w-4 h-4" />
                          </button>

                          <button
                            onClick={(e) => handleDownloadJSON(item.id, e)}
                            disabled={downloadingId === item.id}
                            className="p-2 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 hover:text-cyan-600 dark:hover:text-cyan-400 hover:bg-slate-200 dark:hover:bg-slate-700 transition"
                            title="Download Analysis JSON"
                          >
                            <FileCode className="w-4 h-4" />
                          </button>

                          <Link
                            to={`/results/${item.id}`}
                            className="inline-flex items-center gap-1.5 px-3.5 py-2 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-bold text-xs transition shadow-sm"
                          >
                            <span>Review</span>
                            <ArrowRight className="w-3.5 h-3.5" />
                          </Link>
                        </div>
                      </td>
                    </tr>
                  );
                })
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
