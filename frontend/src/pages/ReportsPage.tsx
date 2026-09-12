import React, { useState, useEffect } from 'react';
import {
  FileText,
  Download,
  CheckCircle2,
  Clock,
  Search,
  Filter,
  ArrowUpRight,
  ShieldAlert,
  Calendar,
  Layers,
  FileCode
} from 'lucide-react';
import { apiService } from '../services/api';
import { ReportItem } from '../types/api';

export const ReportsPage: React.FC = () => {
  const [reports, setReports] = useState<ReportItem[]>([]);
  const [searchTerm, setSearchTerm] = useState('');
  const [filterStatus, setFilterStatus] = useState<string>('all');
  const [downloadingId, setDownloadingId] = useState<string | null>(null);

  useEffect(() => {
    apiService.getReports().then(setReports);
  }, []);

  const handleDownloadPDF = async (analysisId: string) => {
    setDownloadingId(analysisId);
    try {
      await apiService.downloadClinicalReport(analysisId, `dental_caries_clinical_report_${analysisId}.pdf`);
    } catch (e) {
      console.error(e);
    } finally {
      setDownloadingId(null);
    }
  };

  const handleDownloadJSON = async (analysisId: string) => {
    setDownloadingId(analysisId);
    try {
      await apiService.downloadAnalysisJSON(analysisId, `dental_caries_analysis_${analysisId}.json`);
    } catch (e) {
      console.error(e);
    } finally {
      setDownloadingId(null);
    }
  };

  const filteredReports = reports.filter((item) => {
    const matchesSearch =
      item.analysisId.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.filename.toLowerCase().includes(searchTerm.toLowerCase()) ||
      item.patientPseudoId.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter =
      filterStatus === 'all' || item.reviewStatus.toLowerCase().includes(filterStatus.toLowerCase());
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="space-y-8 max-w-7xl mx-auto pb-12">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 text-xs font-semibold mb-2">
            <FileText className="w-3.5 h-3.5" />
            <span>Clinical Documentation & Exports</span>
          </div>
          <h1 className="text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
            Reports & Exports
          </h1>
          <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
            Clinical decision support reports and case data exports for panoramic dental radiograph screenings.
          </p>
        </div>
      </div>

      {/* Clinical AI Summary Banner */}
      <div className="p-6 rounded-3xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
        <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <ShieldAlert className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
          <span>Clinical AI Summary Specification</span>
        </h3>
        <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
          Each generated clinical report provides an official 2-page radiology summary including: raw radiograph panels, AI-assisted caries candidate localization highlights, binary segmentation masks (Background = 0, Caries Candidate = 1), quantitative pixel area measurements, AI-assigned severity staging, and licensed dentist review sign-off.
        </p>
      </div>

      {/* Search & Filter Bar */}
      <div className="flex flex-col sm:flex-row items-center justify-between gap-4">
        <div className="relative w-full sm:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-3" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by Case ID, Patient ID, or File..."
            className="w-full pl-9 pr-3.5 py-2.5 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-xl text-xs text-slate-900 dark:text-slate-100 placeholder-slate-400 focus:outline-none focus:border-cyan-500"
          />
        </div>

        <div className="flex items-center gap-2 w-full sm:w-auto">
          <span className="text-xs text-slate-500 dark:text-slate-400 font-medium">Review Status:</span>
          <select
            value={filterStatus}
            onChange={(e) => setFilterStatus(e.target.value)}
            className="px-3 py-2 bg-white dark:bg-slate-950 border border-slate-300 dark:border-slate-800 rounded-xl text-xs text-slate-800 dark:text-slate-200 focus:outline-none focus:border-cyan-500"
          >
            <option value="all">All Cases</option>
            <option value="verified">Clinically Verified</option>
            <option value="pending">Pending Review</option>
          </select>
        </div>
      </div>

      {/* Reports Table */}
      <div className="bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl overflow-hidden shadow-sm">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs border-collapse">
            <thead>
              <tr className="bg-slate-50 dark:bg-[#121b2d] border-b border-slate-200 dark:border-[#1b2742] text-slate-600 dark:text-slate-400 font-semibold uppercase text-[10px]">
                <th className="py-3.5 px-4">Case ID</th>
                <th className="py-3.5 px-4">Exam Date</th>
                <th className="py-3.5 px-4">Patient Ref</th>
                <th className="py-3.5 px-4">Finding</th>
                <th className="py-3.5 px-4">Stage</th>
                <th className="py-3.5 px-4">Lesions</th>
                <th className="py-3.5 px-4">Review Status</th>
                <th className="py-3.5 px-4 text-right">Available Exports</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100 dark:divide-[#1b2742]">
              {filteredReports.map((rep) => (
                <tr key={rep.id} className="hover:bg-slate-50 dark:hover:bg-[#121b2d]/50 transition">
                  <td className="py-3.5 px-4 font-mono font-bold text-cyan-600 dark:text-cyan-400">
                    {rep.analysisId}
                  </td>
                  <td className="py-3.5 px-4 text-slate-500 dark:text-slate-400 font-mono">
                    {rep.timestamp}
                  </td>
                  <td className="py-3.5 px-4 text-slate-700 dark:text-slate-300 font-medium">
                    {rep.patientPseudoId}
                  </td>
                  <td className="py-3.5 px-4 font-semibold text-slate-800 dark:text-slate-200">
                    {rep.overallFinding}
                  </td>
                  <td className="py-3.5 px-4">
                    <span className="px-2 py-0.5 rounded-full text-[10px] font-semibold bg-cyan-50 dark:bg-cyan-500/15 text-cyan-700 dark:text-cyan-300 border border-cyan-200 dark:border-cyan-500/30">
                      {rep.stageLevel}
                    </span>
                  </td>
                  <td className="py-3.5 px-4 font-mono text-slate-700 dark:text-slate-300">
                    {rep.lesionCount} site(s)
                  </td>
                  <td className="py-3.5 px-4">
                    {rep.reviewStatus === 'Clinically Verified' ? (
                      <span className="inline-flex items-center gap-1 text-[11px] text-emerald-600 dark:text-emerald-400 font-semibold">
                        <CheckCircle2 className="w-3.5 h-3.5" />
                        {rep.reviewStatus}
                      </span>
                    ) : (
                      <span className="inline-flex items-center gap-1 text-[11px] text-amber-600 dark:text-amber-400 font-semibold">
                        <Clock className="w-3.5 h-3.5" />
                        {rep.reviewStatus}
                      </span>
                    )}
                  </td>
                  <td className="py-3.5 px-4 text-right">
                    <div className="inline-flex items-center gap-2">
                      <button
                        onClick={() => handleDownloadPDF(rep.analysisId)}
                        disabled={downloadingId === rep.analysisId}
                        className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-cyan-50 dark:bg-cyan-500/10 text-cyan-700 dark:text-cyan-300 border border-cyan-200 dark:border-cyan-500/30 hover:bg-cyan-600 hover:text-white dark:hover:bg-cyan-500 dark:hover:text-slate-950 font-semibold text-xs transition disabled:opacity-50"
                        title="Download 2-Page Clinical PDF"
                      >
                        <Download className="w-3.5 h-3.5" />
                        <span>PDF</span>
                      </button>

                      <button
                        onClick={() => handleDownloadJSON(rep.analysisId)}
                        disabled={downloadingId === rep.analysisId}
                        className="inline-flex items-center gap-1 px-3 py-1.5 rounded-lg bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300 border border-slate-200 dark:border-slate-700 hover:bg-slate-200 dark:hover:bg-slate-700 font-semibold text-xs transition disabled:opacity-50"
                        title="Download Analysis JSON"
                      >
                        <FileCode className="w-3.5 h-3.5" />
                        <span>JSON</span>
                      </button>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
