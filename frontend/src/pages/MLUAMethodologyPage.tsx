import React from 'react';
import { 
  Binary, 
  Layers, 
  ScanLine, 
  Stethoscope, 
  ShieldCheck, 
  Activity, 
  Sliders, 
  FileText,
  HelpCircle,
  Eye
} from 'lucide-react';
import { PRODUCT_INFO } from '../constants/clinicalMetadata';

export const MLUAMethodologyPage: React.FC = () => {
  return (
    <div className="space-y-8 max-w-5xl mx-auto pb-12">
      {/* Header */}
      <div className="border-b border-slate-200 dark:border-[#1b2742] pb-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 text-xs font-semibold mb-2">
          <Binary className="w-3.5 h-3.5" />
          <span>Scientific Methodology & Image Analysis</span>
        </div>
        <h1 className="text-3xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
          Methods
        </h1>
        <p className="text-slate-500 dark:text-slate-400 text-sm mt-1">
          How the {PRODUCT_INFO.name} system analyzes panoramic radiographs to identify candidate caries regions.
        </p>
      </div>

      {/* 5-Step Clinical Processing Workflow */}
      <div className="space-y-4">
        <h2 className="text-lg font-bold text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <ScanLine className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
          <span>Clinical Radiograph Analysis Pipeline</span>
        </h2>
        <p className="text-xs text-slate-500 dark:text-slate-400">
          The automated screening workflow processes input panoramic images through standardized multi-stage steps:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 pt-2">
          {/* Step 1 */}
          <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
            <div className="w-8 h-8 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 font-bold flex items-center justify-center text-xs">
              01
            </div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">1. Image Standardization</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Standardizes grayscale panoramic radiographs (OPGs), normalizing dynamic contrast ranges and compensating for varying clinical radiographic exposure settings.
            </p>
          </div>

          {/* Step 2 */}
          <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
            <div className="w-8 h-8 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 font-bold flex items-center justify-center text-xs">
              02
            </div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">2. Hierarchical Feature Extraction</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Multi-scale neural network encoders extract hierarchical anatomical features across both macroscopic arch curves and fine interproximal enamel/dentin margins.
            </p>
          </div>

          {/* Step 3 */}
          <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
            <div className="w-8 h-8 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 font-bold flex items-center justify-center text-xs">
              03
            </div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">3. Pixel-Level Segmentation</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Dense feature pyramids fuse multi-resolution representation maps to classify individual pixels into caries candidate regions vs intact dental hard tissues.
            </p>
          </div>

          {/* Step 4 */}
          <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
            <div className="w-8 h-8 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 font-bold flex items-center justify-center text-xs">
              04
            </div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">4. Lesion Localization & Staging</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Discrete contiguous candidate clusters are grouped into numbered lesion sites (Lesion 01, 02) and assigned screening severity stages based on total demineralized area percentage.
            </p>
          </div>

          {/* Step 5 */}
          <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
            <div className="w-8 h-8 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 font-bold flex items-center justify-center text-xs">
              05
            </div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">5. Doctor Verification</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              A licensed dental practitioner inspects the overlay findings, completes the clinical review form, and documents clinical agreement and necessary follow-up examinations.
            </p>
          </div>

          {/* Step 6 */}
          <div className="p-5 rounded-2xl bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] shadow-sm space-y-3">
            <div className="w-8 h-8 rounded-xl bg-cyan-50 dark:bg-cyan-500/10 text-cyan-600 dark:text-cyan-400 font-bold flex items-center justify-center text-xs">
              06
            </div>
            <h3 className="font-bold text-sm text-slate-900 dark:text-slate-100">6. Clinical Documentation</h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Generates a verified 2-page Clinical Radiology Screening & Candidate Localization Report ready for electronic health record (EHR) archiving and patient consultation.
            </p>
          </div>
        </div>
      </div>

      {/* Clinical Decision Support Guidelines */}
      <div className="bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl p-6 sm:p-8 space-y-4 shadow-sm">
        <h3 className="font-bold text-base text-slate-900 dark:text-slate-100 flex items-center gap-2">
          <Stethoscope className="w-5 h-5 text-cyan-600 dark:text-cyan-400" />
          <span>Understanding Caries Screening Outputs</span>
        </h3>
        <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
          The system highlights suspicious areas of radiolucency. Because panoramic radiography compresses three-dimensional anatomy into a single curved focal plane, the following clinical guidelines must be considered:
        </p>

        <ul className="space-y-2 text-xs text-slate-700 dark:text-slate-300 list-disc list-inside">
          <li><strong>Interproximal Surfaces:</strong> Subtle enamel demineralization on adjacent premolar and molar contacts should be verified using bitewing radiographs.</li>
          <li><strong>Cervical Regions:</strong> The cervical narrowing of tooth crowns between enamel and alveolar bone naturally creates radiolucent &quot;burnout&quot; zones that require visual-tactile differentiation.</li>
          <li><strong>Restorative Margins:</strong> Resin composites without heavy radiopaque fillers may appear radiolucent under automated segmentation and should be cross-referenced with dental treatment history.</li>
        </ul>
      </div>

      {/* Technical Specifications for Transparency */}
      <div className="p-6 rounded-3xl bg-slate-50 dark:bg-[#121b2d]/60 border border-slate-200 dark:border-[#1b2742] space-y-4">
        <h4 className="font-bold text-xs uppercase tracking-wider text-slate-500 dark:text-slate-400 flex items-center gap-2">
          <Activity className="w-4 h-4 text-cyan-600 dark:text-cyan-400" />
          <span>System Architecture & Specifications</span>
        </h4>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 text-xs font-mono">
          <div className="p-3 bg-white dark:bg-[#070a12] rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-[10px] text-slate-400 block font-sans">BACKBONE</span>
            <span className="font-semibold text-slate-800 dark:text-slate-200">Feature Pyramid Network</span>
          </div>
          <div className="p-3 bg-white dark:bg-[#070a12] rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-[10px] text-slate-400 block font-sans">RESOLUTION LEVELS</span>
            <span className="font-semibold text-slate-800 dark:text-slate-200">Multi-Scale Pyramidal</span>
          </div>
          <div className="p-3 bg-white dark:bg-[#070a12] rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-[10px] text-slate-400 block font-sans">INPUT MODALITY</span>
            <span className="font-semibold text-slate-800 dark:text-slate-200">Dental Panoramic OPG</span>
          </div>
          <div className="p-3 bg-white dark:bg-[#070a12] rounded-xl border border-slate-200 dark:border-slate-800">
            <span className="text-[10px] text-slate-400 block font-sans">CLASSIFICATION TARGET</span>
            <span className="font-semibold text-slate-800 dark:text-slate-200">Caries vs Intact Tissue</span>
          </div>
        </div>
      </div>

      {/* Technical Implementation Section */}
      <div className="bg-white dark:bg-[#0d1322] border border-slate-200 dark:border-[#1b2742] rounded-3xl p-6 sm:p-8 space-y-6 shadow-sm">
        <div>
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-50 dark:bg-cyan-500/10 border border-cyan-200 dark:border-cyan-500/30 text-cyan-700 dark:text-cyan-400 text-xs font-semibold mb-2">
            <Layers className="w-3.5 h-3.5" />
            <span>Under the Hood</span>
          </div>
          <h3 className="text-xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
            Technical Implementation & Training Protocol
          </h3>
          <p className="text-slate-500 dark:text-slate-400 text-xs mt-1">
            Mathematical foundations, loss functions, and uncertainty quantification mechanisms powering the segmentation engine.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Architecture & Multi-Scale Heads */}
          <div className="space-y-3 p-5 rounded-2xl bg-slate-50 dark:bg-[#121b2d]/40 border border-slate-200 dark:border-[#1b2742]">
            <h4 className="font-bold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              Multi-Scale Pyramidal Architecture
            </h4>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              The model utilizes a hierarchical encoder coupled with a lateral Feature Pyramid Network (FPN) decoder. Four multi-scale auxiliary heads (H1 to H4) produce intermediate segmentation predictions at resolutions of 1/8, 1/4, 1/2, and full scale, enforcing progressive gradient flow:
            </p>
            <div className="p-3 bg-white dark:bg-[#070a12] rounded-xl border border-slate-200 dark:border-slate-800 font-mono text-[11px] text-cyan-600 dark:text-cyan-400 leading-relaxed">
              L_sup = Σ (α_k * [L_BCE(P_k, Y) + L_SoftDice(P_k, Y)])
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Where α = [0.1, 0.2, 0.3, 0.4] progressively weights deeper resolution stages.
            </p>
          </div>

          {/* Uncertainty Estimation */}
          <div className="space-y-3 p-5 rounded-2xl bg-slate-50 dark:bg-[#121b2d]/40 border border-slate-200 dark:border-[#1b2742]">
            <h4 className="font-bold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              Monte Carlo Uncertainty Estimation (T=8)
            </h4>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              To quantify epistemic ambiguity in low-contrast radiographic zones (such as cervical burnout or restorative boundaries), the framework executes T=8 stochastic forward passes under active dropout:
            </p>
            <div className="p-3 bg-white dark:bg-[#070a12] rounded-xl border border-slate-200 dark:border-slate-800 font-mono text-[11px] text-cyan-600 dark:text-cyan-400 leading-relaxed">
              μ(x) = (1/T) Σ ŷ_t(x), &nbsp; σ²(x) = (1/T) Σ [ŷ_t(x) - μ(x)]²
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Pixel-wise variance σ²(x) gates unconfident pseudo-labels to prevent noisy gradient propagation.
            </p>
          </div>

          {/* Loss Formulation */}
          <div className="space-y-3 p-5 rounded-2xl bg-slate-50 dark:bg-[#121b2d]/40 border border-slate-200 dark:border-[#1b2742]">
            <h4 className="font-bold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              Composite Loss Formulation
            </h4>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Soft Dice loss directly optimizes spatial overlap against extreme class imbalance:
            </p>
            <div className="p-3 bg-white dark:bg-[#070a12] rounded-xl border border-slate-200 dark:border-slate-800 font-mono text-[11px] text-cyan-600 dark:text-cyan-400 leading-relaxed">
              L_SoftDice = 1 - [2 * Σ(P_i * Y_i) + ε] / [Σ(P_i²) + Σ(Y_i²) + ε]
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Total objective balances supervised overlap and unsupervised consistency: L_total = L_sup + λ(t) * L_unsup.
            </p>
          </div>

          {/* Semi-Supervised Learning Protocol */}
          <div className="space-y-3 p-5 rounded-2xl bg-slate-50 dark:bg-[#121b2d]/40 border border-slate-200 dark:border-[#1b2742]">
            <h4 className="font-bold text-sm text-slate-900 dark:text-slate-100 flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              Semi-Supervised Learning (10% Labeled)
            </h4>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Unlabeled panoramic cases are trained using uncertainty-weighted consistency regularization:
            </p>
            <div className="p-3 bg-white dark:bg-[#070a12] rounded-xl border border-slate-200 dark:border-slate-800 font-mono text-[11px] text-cyan-600 dark:text-cyan-400 leading-relaxed">
              L_unsup = (1/|U|) Σ exp(-σ²(x)) * || f(x; θ) - f(x̃; θ') ||²
            </div>
            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Exponential weighting downweights high-variance artifacts while amplifying confident structural features.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};
