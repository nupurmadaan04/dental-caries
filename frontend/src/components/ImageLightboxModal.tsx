import React, { useState, useEffect } from 'react';
import { X, ZoomIn, ZoomOut, RotateCcw, Layers, Maximize2 } from 'lucide-react';
import { LesionFinding } from '../types/api';

interface LightboxProps {
  isOpen: boolean;
  onClose: () => void;
  originalUrl: string;
  overlayUrl?: string;
  binaryMaskUrl?: string;
  findings: LesionFinding[];
  initialLayer?: 'overlay' | 'binary' | 'original';
}

export const ImageLightboxModal: React.FC<LightboxProps> = ({
  isOpen,
  onClose,
  originalUrl,
  overlayUrl,
  binaryMaskUrl,
  findings,
  initialLayer = 'overlay',
}) => {
  const [zoom, setZoom] = useState(1);
  const [activeLayer, setActiveLayer] = useState<'overlay' | 'binary' | 'original'>(initialLayer);
  const [showBoxes, setShowBoxes] = useState(true);

  useEffect(() => {
    if (isOpen && initialLayer) {
      setActiveLayer(initialLayer);
      setZoom(1);
    }
  }, [isOpen, initialLayer]);

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-md p-4 animate-fade-in">
      {/* Container */}
      <div className="relative w-full max-w-6xl h-[88vh] bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl flex flex-col overflow-hidden shadow-2xl">
        {/* Header Bar */}
        <div className="h-16 px-6 border-b border-slate-200 dark:border-[#1b2742] flex items-center justify-between bg-slate-50 dark:bg-[#121b2d]">
          <div className="flex items-center gap-3">
            <Maximize2 className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
            <span className="font-bold text-slate-900 dark:text-slate-100 text-sm">
              Panoramic Radiograph Inspection Viewer
            </span>
          </div>

          {/* Controls */}
          <div className="flex items-center gap-3">
            {/* Layer Switcher */}
            <div className="flex bg-slate-200 dark:bg-[#070a12] p-1 rounded-xl border border-slate-300 dark:border-[#1b2742]">
              <button
                onClick={() => setActiveLayer('overlay')}
                className={`px-3 py-1 text-xs rounded-lg font-medium transition-all ${
                  activeLayer === 'overlay' ? 'bg-cyan-600 text-white font-bold' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Caries Overlay
              </button>
              <button
                onClick={() => setActiveLayer('binary')}
                className={`px-3 py-1 text-xs rounded-lg font-medium transition-all ${
                  activeLayer === 'binary' ? 'bg-slate-800 text-white font-bold' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Binary Mask
              </button>
              <button
                onClick={() => setActiveLayer('original')}
                className={`px-3 py-1 text-xs rounded-lg font-medium transition-all ${
                  activeLayer === 'original' ? 'bg-slate-700 text-white font-bold' : 'text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200'
                }`}
              >
                Original OPG
              </button>
            </div>

            {/* Bounding Box Toggle */}
            <button
              onClick={() => setShowBoxes(!showBoxes)}
              className={`p-2 rounded-xl border transition-all ${
                showBoxes ? 'bg-cyan-100 dark:bg-cyan-500/20 border-cyan-300 dark:border-cyan-500/40 text-cyan-700 dark:text-cyan-300' : 'bg-slate-100 dark:bg-[#070a12] border-slate-200 dark:border-[#1b2742] text-slate-500'
              }`}
              title="Toggle Lesion Markers"
            >
              <Layers className="w-4 h-4" />
            </button>

            {/* Zoom Controls */}
            <div className="flex items-center bg-slate-200 dark:bg-[#070a12] rounded-xl border border-slate-300 dark:border-[#1b2742] p-1">
              <button
                onClick={() => setZoom(Math.max(1, zoom - 0.25))}
                className="p-1.5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 rounded"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>
              <span className="text-xs font-mono text-slate-700 dark:text-slate-300 px-2">{zoom.toFixed(2)}x</span>
              <button
                onClick={() => setZoom(Math.min(3.5, zoom + 0.25))}
                className="p-1.5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 rounded"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>
              <button
                onClick={() => setZoom(1)}
                className="p-1.5 text-slate-600 dark:text-slate-400 hover:text-slate-900 dark:hover:text-slate-200 border-l border-slate-300 dark:border-[#1b2742] ml-1"
                title="Reset Zoom"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>

            {/* Close */}
            <button
              onClick={onClose}
              className="p-2 rounded-xl text-slate-400 hover:text-slate-700 dark:hover:text-slate-100 hover:bg-slate-200 dark:hover:bg-slate-800 transition-all"
            >
              <X className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Viewer Viewport */}
        <div className="flex-1 relative overflow-auto bg-slate-900 dark:bg-slate-950 flex items-center justify-center p-6 select-none">
          <div
            className="relative transition-transform duration-100 max-w-full max-h-full rounded-2xl overflow-hidden border border-slate-700 dark:border-slate-800 shadow-2xl flex items-center justify-center"
            style={{ transform: `scale(${zoom})`, transformOrigin: 'center center' }}
          >
            {activeLayer === 'binary' ? (
              <div className="relative max-w-full max-h-[72vh] flex items-center justify-center bg-black">
                <img
                  src={binaryMaskUrl || '/samples/opg_mask.png'}
                  alt="Binary Segmentation Mask"
                  className="max-h-[70vh] w-auto object-contain bg-black"
                />
                <span className="absolute bottom-3 left-3 bg-black/80 px-2 py-1 rounded text-[10px] font-mono text-slate-400 border border-slate-800">
                  Binary Mask (Caries: White, Background: Black)
                </span>
              </div>
            ) : activeLayer === 'original' ? (
              <div className="relative max-w-full max-h-[72vh] flex items-center justify-center bg-black">
                <img
                  src={originalUrl || '/samples/opg_sample.png'}
                  alt="Original Panoramic Radiograph"
                  className="max-h-[70vh] w-auto object-contain"
                />
                <span className="absolute bottom-3 left-3 bg-black/80 px-2 py-1 rounded text-[10px] font-mono text-slate-400 border border-slate-800">
                  Original Standardized Radiograph (OPG)
                </span>
              </div>
            ) : (
              <div className="relative max-w-full max-h-[72vh] flex items-center justify-center bg-black">
                <img
                  src={overlayUrl || originalUrl || '/samples/opg_sample.png'}
                  alt="Radiograph with AI Overlay"
                  className="max-h-[70vh] w-auto object-contain"
                />

                {/* Real Lesion Bounding Boxes & Pins on Overlay */}
                {showBoxes &&
                  findings.map((f, idx) => {
                    const [origX, origY, origW, origH] = f.bbox;
                    // Proportional coordinates assuming 600x300 or native scale
                    return (
                      <div
                        key={f.id}
                        className="absolute border-2 border-cyan-400 rounded bg-cyan-400/25 shadow-[0_0_12px_rgba(34,211,238,0.7)] flex flex-col justify-start p-0.5 pointer-events-auto animate-pulse"
                        style={{
                          left: `${(origX / 600) * 100}%`,
                          top: `${(origY / 300) * 100}%`,
                          width: `${Math.max(6, (origW / 600) * 100)}%`,
                          height: `${Math.max(6, (origH / 300) * 100)}%`,
                        }}
                      >
                        <span className="text-[9px] font-black font-mono bg-cyan-400 text-slate-950 px-1 rounded-sm w-fit -mt-4 shadow-sm">
                          {f.lesionNumber || `L${idx + 1}`}: {(f.confidence * 100).toFixed(0)}%
                        </span>
                      </div>
                    );
                  })}

                <span className="absolute bottom-3 left-3 bg-black/80 px-2 py-1 rounded text-[10px] font-mono text-cyan-400 border border-cyan-500/30">
                  AI Candidate Overlay ({findings.length} Site{findings.length === 1 ? '' : 's'})
                </span>
              </div>
            )}
          </div>
        </div>

        {/* Footer Info */}
        <div className="h-12 px-6 border-t border-slate-200 dark:border-[#1b2742] bg-slate-50 dark:bg-[#121b2d] flex items-center justify-between text-xs text-slate-500 dark:text-slate-400">
          <div>
            Identified <span className="text-cyan-600 dark:text-cyan-400 font-bold">{findings.length} candidate lesion region(s)</span> for clinical review
          </div>
          <div className="flex items-center gap-4 text-[11px]">
            <span>Background: 0 (Black)</span>
            <span>|</span>
            <span>Caries Candidate: 1 (White / Cyan Overlay)</span>
          </div>
        </div>
      </div>
    </div>
  );
};
