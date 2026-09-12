import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  UploadCloud,
  FileImage,
  CheckCircle2,
  AlertCircle,
  Sliders,
  X,
  Stethoscope,
  ChevronDown,
  ChevronUp
} from 'lucide-react';
import { apiService } from '../services/api';

export const NewAnalysisPage: React.FC = () => {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [threshold, setThreshold] = useState<number>(0.50);
  const [showSettings, setShowSettings] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [currentStepIndex, setCurrentStepIndex] = useState(0);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  const clinicalSteps = [
    'Preparing radiograph',
    'Examining dental structures',
    'Identifying suspicious caries regions',
    'Mapping detected regions',
    'Preparing clinical findings',
  ];

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setErrorMsg(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      const file = e.dataTransfer.files[0];
      setSelectedFile(file);
      setErrorMsg(null);
      const url = URL.createObjectURL(file);
      setPreviewUrl(url);
    }
  };

  const handleStartAnalysis = async () => {
    if (!selectedFile) {
      setErrorMsg('Please select or upload a panoramic radiograph before analyzing.');
      return;
    }

    setIsAnalyzing(true);
    setErrorMsg(null);
    setCurrentStepIndex(0);

    const interval = setInterval(() => {
      setCurrentStepIndex((prev) => {
        if (prev < clinicalSteps.length - 1) {
          return prev + 1;
        }
        return prev;
      });
    }, 600);

    try {
      const result = await apiService.analyzeRadiograph(selectedFile, threshold);
      clearInterval(interval);
      navigate(`/results/${result.id}`);
    } catch (err: any) {
      clearInterval(interval);
      setIsAnalyzing(false);
      setErrorMsg('Analysis service encountered an issue. Please try again.');
    }
  };

  const handleLoadSample = () => {
    const sampleUrl = '/samples/opg_sample.png';
    setPreviewUrl(sampleUrl);
    const blob = new Blob(['sample-opg'], { type: 'image/png' });
    const file = new File([blob], 'clinical_opg_panoramic_case01.png', { type: 'image/png' });
    setSelectedFile(file);
    setErrorMsg(null);
  };

  const handleRemoveImage = () => {
    setSelectedFile(null);
    setPreviewUrl(null);
    setErrorMsg(null);
  };

  return (
    <div className="space-y-8 max-w-4xl mx-auto pb-16">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <h1 className="text-2xl sm:text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
          New Panoramic Analysis
        </h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          Upload a panoramic dental radiograph for AI-assisted caries screening and clinical review.
        </p>
      </div>

      {errorMsg && (
        <div className="p-4 rounded-2xl bg-rose-50 dark:bg-rose-950/40 border border-rose-200 dark:border-rose-900/50 text-rose-800 dark:text-rose-300 text-xs flex items-center gap-3">
          <AlertCircle className="w-4 h-4 shrink-0 text-rose-600 dark:text-rose-400" />
          <span>{errorMsg}</span>
        </div>
      )}

      {/* Main Upload Panel */}
      <div className="bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl p-6 sm:p-8 shadow-sm space-y-6">
        <div>
          <h2 className="text-base font-bold text-slate-900 dark:text-slate-100">
            Upload Panoramic Radiograph
          </h2>
          <p className="text-xs text-slate-500 dark:text-slate-400 mt-0.5">
            Upload a clear dental panoramic radiograph in PNG, JPG, JPEG, TIFF, BMP or DICOM format.
          </p>
        </div>

        {/* Drag & Drop Area */}
        <div
          onDragOver={(e) => e.preventDefault()}
          onDrop={handleDrop}
          className={`relative border-2 border-dashed rounded-2xl p-8 text-center transition-all ${
            selectedFile
              ? 'border-cyan-500/50 bg-cyan-50/10 dark:bg-cyan-950/10'
              : 'border-slate-300 dark:border-slate-700/80 hover:border-cyan-500 dark:hover:border-cyan-500 bg-slate-50/50 dark:bg-[#121b2d]/50'
          }`}
        >
          {previewUrl ? (
            <div className="space-y-4">
              <div className="relative rounded-xl overflow-hidden border border-slate-200 dark:border-slate-800 bg-black max-w-2xl mx-auto shadow-md">
                <img
                  src={previewUrl}
                  alt="Radiograph Preview"
                  className="w-full max-h-80 object-contain mx-auto"
                />
              </div>

              <div className="text-xs text-slate-600 dark:text-slate-300 font-medium flex items-center justify-center gap-2">
                <CheckCircle2 className="w-4 h-4 text-emerald-500" />
                <span className="font-semibold">{selectedFile?.name || 'sachin.png'}</span>
                <span className="text-slate-400">
                  ({(selectedFile?.size ? (selectedFile.size / 1024).toFixed(1) : '840')} KB)
                </span>
              </div>
            </div>
          ) : (
            <div className="space-y-4 py-8">
              <div className="w-14 h-14 rounded-2xl bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 flex items-center justify-center text-cyan-600 dark:text-cyan-400 mx-auto">
                <UploadCloud className="w-7 h-7" />
              </div>

              <div className="space-y-1">
                <p className="text-sm font-semibold text-slate-800 dark:text-slate-200">
                  Drag & drop panoramic radiograph here
                </p>
                <p className="text-xs text-slate-400">or</p>
              </div>

              <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-1">
                <label className="px-5 py-2.5 rounded-xl bg-cyan-600 hover:bg-cyan-500 text-white font-semibold text-xs cursor-pointer shadow-md shadow-cyan-600/20 transition">
                  <span>Browse Files</span>
                  <input
                    type="file"
                    accept="image/png,image/jpeg,image/tiff,image/bmp,image/dicom,.dcm"
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                </label>

                <button
                  type="button"
                  onClick={handleLoadSample}
                  className="px-4 py-2.5 rounded-xl bg-slate-100 dark:bg-[#121b2d] border border-slate-200 dark:border-[#1b2742] hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 text-xs font-semibold transition"
                >
                  Load Example OPG Scan
                </button>
              </div>

              <p className="text-[11px] text-slate-400 dark:text-slate-500 pt-3">
                Supported formats: PNG, JPG, JPEG, TIFF, BMP, DICOM
              </p>
            </div>
          )}
        </div>

        {/* Action Controls */}
        {selectedFile && (
          <div className="space-y-4 pt-2">
            <div className="flex flex-col sm:flex-row items-center gap-3">
              <button
                onClick={handleStartAnalysis}
                disabled={isAnalyzing}
                className="w-full sm:flex-1 py-3 px-6 rounded-xl bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 text-white font-bold text-xs shadow-md shadow-cyan-600/20 transition flex items-center justify-center gap-2"
              >
                {isAnalyzing ? (
                  <>
                    <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                    <span>Analyzing Radiograph...</span>
                  </>
                ) : (
                  <>
                    <Stethoscope className="w-4 h-4" />
                    <span>Analyze Radiograph</span>
                  </>
                )}
              </button>

              <button
                onClick={handleRemoveImage}
                disabled={isAnalyzing}
                className="w-full sm:w-auto py-3 px-5 rounded-xl bg-slate-100 dark:bg-[#121b2d] border border-slate-200 dark:border-[#1b2742] hover:bg-slate-200 dark:hover:bg-slate-800 text-slate-700 dark:text-slate-300 font-semibold text-xs transition"
              >
                Remove Image
              </button>
            </div>

            {/* Collapsible Advanced Settings */}
            <div className="border-t border-slate-200 dark:border-[#1b2742] pt-3">
              <button
                type="button"
                onClick={() => setShowSettings(!showSettings)}
                className="flex items-center gap-2 text-xs font-semibold text-slate-500 hover:text-slate-800 dark:hover:text-slate-200 transition"
              >
                <Sliders className="w-3.5 h-3.5" />
                <span>Analysis Settings</span>
                {showSettings ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
              </button>

              {showSettings && (
                <div className="mt-4 p-4 rounded-2xl bg-slate-50 dark:bg-[#121b2d] border border-slate-200 dark:border-[#1b2742] space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-700 dark:text-slate-300">
                      Detection Sensitivity Threshold
                    </span>
                    <span className="text-xs font-mono font-bold text-cyan-600 dark:text-cyan-400">
                      {threshold.toFixed(2)} {threshold === 0.50 && '(Optimal)'}
                    </span>
                  </div>
                  <input
                    type="range"
                    min="0.10"
                    max="0.90"
                    step="0.05"
                    value={threshold}
                    onChange={(e) => setThreshold(parseFloat(e.target.value))}
                    className="w-full accent-cyan-600 h-2 bg-slate-200 dark:bg-slate-800 rounded-lg cursor-pointer"
                  />
                  <div className="flex justify-between text-[10px] text-slate-400">
                    <span>High Sensitivity (0.10)</span>
                    <span className="font-semibold text-cyan-600 dark:text-cyan-400">Optimal Clinical (0.50)</span>
                    <span>High Specificity (0.90)</span>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Analysis Processing State */}
      {isAnalyzing && (
        <div className="p-6 sm:p-8 rounded-3xl bg-white dark:bg-slate-900 text-slate-900 dark:text-white border border-slate-200 dark:border-slate-800 space-y-5 animate-fade-in shadow-xl dark:shadow-2xl">
          <div className="flex items-center justify-between">
            <h4 className="font-bold text-sm text-cyan-600 dark:text-cyan-400 flex items-center gap-2">
              <div className="w-2.5 h-2.5 rounded-full bg-cyan-500 animate-ping" />
              <span>Analyzing Panoramic Radiograph</span>
            </h4>
            <span className="text-xs font-mono text-slate-500 dark:text-slate-400">
              Stage {currentStepIndex + 1} of {clinicalSteps.length}
            </span>
          </div>

          <div className="w-full bg-slate-100 dark:bg-slate-800 h-2 rounded-full overflow-hidden">
            <div
              className="bg-cyan-500 h-full transition-all duration-300 rounded-full"
              style={{ width: `${((currentStepIndex + 1) / clinicalSteps.length) * 100}%` }}
            />
          </div>

          <div className="space-y-2.5 pt-1">
            {clinicalSteps.map((step, idx) => (
              <div
                key={idx}
                className={`flex items-center gap-3 text-xs transition-colors ${
                  idx === currentStepIndex
                    ? 'text-cyan-600 dark:text-cyan-300 font-bold'
                    : idx < currentStepIndex
                    ? 'text-emerald-600 dark:text-emerald-400 font-medium'
                    : 'text-slate-400 dark:text-slate-500'
                }`}
              >
                {idx < currentStepIndex ? (
                  <CheckCircle2 className="w-4 h-4 text-emerald-600 dark:text-emerald-400 shrink-0" />
                ) : idx === currentStepIndex ? (
                  <div className="w-4 h-4 border-2 border-cyan-500 border-t-transparent rounded-full animate-spin shrink-0" />
                ) : (
                  <div className="w-4 h-4 rounded-full border border-slate-300 dark:border-slate-700 shrink-0" />
                )}
                <span>{step}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
};

