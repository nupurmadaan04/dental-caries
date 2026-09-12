import React from 'react';
import { 
  ScanLine, 
  Layers, 
  Stethoscope, 
  Activity, 
  Cpu, 
  SlidersHorizontal,
  CheckCircle2,
  GitBranch,
  ShieldCheck
} from 'lucide-react';
import { PRODUCT_INFO } from '../constants/clinicalMetadata';

export const MLUAMethodologyPage: React.FC = () => {
  return (
    <div className="space-y-10 max-w-6xl mx-auto pb-16 animate-fade-in text-slate-900 dark:text-slate-100">
      {/* Top Header Section */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-600 dark:text-cyan-400 text-xs font-semibold shadow-sm">
          <SlidersHorizontal className="w-3.5 h-3.5" />
          <span>Scientific Methodology & Image Analysis</span>
        </div>
        
        <h1 className="text-3xl sm:text-4xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
          Methods
        </h1>
        
        <p className="text-slate-600 dark:text-slate-400 text-sm sm:text-base leading-relaxed max-w-3xl">
          How the Dental Caries Clinical AI system analyzes panoramic radiographs to identify candidate caries regions using the validated EXP-MLUA-003 ResNet-34 + FPN semi-supervised segmentation framework.
        </p>
      </div>

      {/* Section 1: Clinical Radiograph Analysis Pipeline */}
      <div className="space-y-4">
        <div className="flex items-center gap-2.5 text-cyan-600 dark:text-cyan-400 font-bold text-lg">
          <ScanLine className="w-5 h-5" />
          <h2 className="text-slate-900 dark:text-slate-100 text-lg font-bold">
            Clinical Radiograph Analysis Pipeline
          </h2>
        </div>
        
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400">
          The automated screening workflow processes input panoramic images through standardized multi-stage steps:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5 pt-2">
          {/* Step 01 */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md hover:border-cyan-500/40 transition-all space-y-3">
            <span className="text-xs font-mono font-bold text-cyan-600 dark:text-cyan-400 block">
              01
            </span>
            <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
              1. Image Standardization
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Standardizes grayscale panoramic radiographs (OPGs), normalizing dynamic contrast ranges and compensating for varying clinical radiographic exposure settings.
            </p>
          </div>

          {/* Step 02 */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md hover:border-cyan-500/40 transition-all space-y-3">
            <span className="text-xs font-mono font-bold text-cyan-600 dark:text-cyan-400 block">
              02
            </span>
            <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
              2. Hierarchical Feature Extraction
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Multi-scale neural network encoders extract hierarchical anatomical features across both macroscopic arch curves and fine interproximal enamel/dentin margins.
            </p>
          </div>

          {/* Step 03 */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md hover:border-cyan-500/40 transition-all space-y-3">
            <span className="text-xs font-mono font-bold text-cyan-600 dark:text-cyan-400 block">
              03
            </span>
            <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
              3. Pixel-Level Segmentation
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Dense feature pyramids fuse multi-resolution representation maps to classify individual pixels into caries candidate regions vs intact dental hard tissues.
            </p>
          </div>

          {/* Step 04 */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md hover:border-cyan-500/40 transition-all space-y-3">
            <span className="text-xs font-mono font-bold text-cyan-600 dark:text-cyan-400 block">
              04
            </span>
            <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
              4. Lesion Localization & Staging
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Discrete contiguous candidate clusters are grouped into numbered lesion sites (L1, L2, L3) and assigned screening severity stages based on total demineralized area percentage.
            </p>
          </div>

          {/* Step 05 */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md hover:border-cyan-500/40 transition-all space-y-3">
            <span className="text-xs font-mono font-bold text-cyan-600 dark:text-cyan-400 block">
              05
            </span>
            <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
              5. Doctor Verification
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              A licensed dental practitioner inspects the overlay findings, completes the clinical review form, and documents clinical agreement and necessary follow-up examinations.
            </p>
          </div>

          {/* Step 06 */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md hover:border-cyan-500/40 transition-all space-y-3">
            <span className="text-xs font-mono font-bold text-cyan-600 dark:text-cyan-400 block">
              06
            </span>
            <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
              6. Clinical Documentation
            </h3>
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Generates a verified 2-page Clinical Radiology Screening & Candidate Localization Report ready for electronic health record (EHR) archiving and patient consultation.
            </p>
          </div>
        </div>
      </div>

      {/* Section 2: Understanding Caries Screening Outputs */}
      <div className="p-6 sm:p-8 rounded-3xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md space-y-4">
        <div className="flex items-center gap-2.5 text-cyan-600 dark:text-cyan-400">
          <Stethoscope className="w-5 h-5" />
          <h3 className="font-bold text-base sm:text-lg text-slate-900 dark:text-slate-100">
            Understanding Caries Screening Outputs
          </h3>
        </div>
        
        <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400 leading-relaxed">
          The system highlights suspicious areas of radiolucency. Because panoramic radiography compresses three-dimensional anatomy into a single curved focal plane, the following clinical guidelines must be considered:
        </p>

        <ul className="space-y-2.5 text-xs sm:text-sm text-slate-700 dark:text-slate-300 list-disc list-inside">
          <li>
            <strong className="text-slate-900 dark:text-slate-100">Interproximal Surfaces:</strong> Subtle enamel demineralization on adjacent premolar and molar contacts should be verified using bitewing radiographs.
          </li>
          <li>
            <strong className="text-slate-900 dark:text-slate-100">Cervical Regions:</strong> The cervical narrowing of tooth crowns between enamel and alveolar bone naturally creates radiolucent &quot;burnout&quot; zones that require visual-tactile differentiation.
          </li>
          <li>
            <strong className="text-slate-900 dark:text-slate-100">Restorative Margins:</strong> Resin composites without heavy radiopaque fillers may appear radiolucent under automated segmentation and should be cross-referenced with dental treatment history.
          </li>
        </ul>
      </div>

      {/* Section 3: SYSTEM ARCHITECTURE & SPECIFICATIONS */}
      <div className="space-y-4">
        <div className="flex items-center gap-2 text-cyan-600 dark:text-cyan-400">
          <Activity className="w-4 h-4" />
          <h3 className="font-bold text-xs uppercase tracking-wider text-slate-700 dark:text-slate-300">
            SYSTEM ARCHITECTURE & SPECIFICATIONS
          </h3>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="p-4 bg-white dark:bg-[#0c1424] rounded-2xl border border-slate-200 dark:border-[#1b2b4d] shadow-sm space-y-1">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans">
              BACKBONE
            </span>
            <span className="font-bold text-xs sm:text-sm text-slate-900 dark:text-slate-100 font-mono">
              ResNet-34 + Lateral FPN
            </span>
          </div>

          <div className="p-4 bg-white dark:bg-[#0c1424] rounded-2xl border border-slate-200 dark:border-[#1b2b4d] shadow-sm space-y-1">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans">
              RESOLUTION LEVELS
            </span>
            <span className="font-bold text-xs sm:text-sm text-slate-900 dark:text-slate-100 font-mono">
              Multi-Scale Pyramidal (4 Aux Heads)
            </span>
          </div>

          <div className="p-4 bg-white dark:bg-[#0c1424] rounded-2xl border border-slate-200 dark:border-[#1b2b4d] shadow-sm space-y-1">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans">
              INPUT MODALITY
            </span>
            <span className="font-bold text-xs sm:text-sm text-slate-900 dark:text-slate-100 font-mono">
              Dental Panoramic OPG (768×1536)
            </span>
          </div>

          <div className="p-4 bg-white dark:bg-[#0c1424] rounded-2xl border border-slate-200 dark:border-[#1b2b4d] shadow-sm space-y-1">
            <span className="text-[10px] text-slate-400 uppercase tracking-wider block font-sans">
              CLASSIFICATION TARGET
            </span>
            <span className="font-bold text-xs sm:text-sm text-cyan-600 dark:text-cyan-400 font-mono">
              Caries vs Intact Tissue (τ = 0.50)
            </span>
          </div>
        </div>
      </div>

      {/* Section 4: Technical Implementation & Training Protocol */}
      <div className="space-y-6 pt-2">
        <div className="space-y-2">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-cyan-500/10 border border-cyan-500/30 text-cyan-600 dark:text-cyan-400 text-xs font-semibold">
            <Layers className="w-3.5 h-3.5" />
            <span>Under the Hood</span>
          </div>

          <h2 className="text-xl sm:text-2xl font-extrabold text-slate-900 dark:text-slate-100 tracking-tight">
            Technical Implementation & Training Protocol
          </h2>

          <p className="text-xs sm:text-sm text-slate-600 dark:text-slate-400">
            Mathematical foundations, loss functions, and uncertainty quantification mechanisms powering the segmentation engine.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Card 1: Multi-Scale Pyramidal Architecture */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md space-y-3.5 hover:border-cyan-500/40 transition-all">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
                Multi-Scale Pyramidal Architecture
              </h3>
            </div>
            
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              The model utilizes a hierarchical encoder coupled with a lateral Feature Pyramid Network (FPN) decoder. Four multi-scale auxiliary heads (H1 to H4) produce intermediate segmentation predictions at resolutions of 1/8, 1/4, 1/2, and full scale, enforcing progressive gradient flow:
            </p>

            <div className="p-3.5 bg-slate-50 dark:bg-[#060a14] rounded-xl border border-slate-200 dark:border-[#16223b] font-mono text-xs text-cyan-700 dark:text-cyan-400 text-center shadow-inner overflow-x-auto">
              <code>L_sup = Σ (α_k * [L_BCE(P_k, Y) + L_SoftDice(P_k, Y)])</code>
            </div>

            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Where α = [0.1, 0.2, 0.3, 0.4] progressively weights deeper resolution stages.
            </p>
          </div>

          {/* Card 2: Monte Carlo Uncertainty Estimation */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md space-y-3.5 hover:border-cyan-500/40 transition-all">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
                Monte Carlo Uncertainty Estimation (T=8)
              </h3>
            </div>
            
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              To quantify epistemic ambiguity in low-contrast radiographic zones (such as cervical burnout or restorative boundaries), the framework executes T=8 stochastic forward passes under active dropout:
            </p>

            <div className="p-3.5 bg-slate-50 dark:bg-[#060a14] rounded-xl border border-slate-200 dark:border-[#16223b] font-mono text-xs text-cyan-700 dark:text-cyan-400 text-center shadow-inner overflow-x-auto">
              <code>μ(x) = (1/T) Σ ŷ_t(x), &nbsp;&nbsp; σ²(x) = (1/T) Σ [ŷ_t(x) - μ(x)]²</code>
            </div>

            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Pixel-wise variance σ²(x) gates unconfident pseudo-labels to prevent noisy gradient propagation.
            </p>
          </div>

          {/* Card 3: Composite Loss Formulation */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md space-y-3.5 hover:border-cyan-500/40 transition-all">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
                Composite Loss Formulation
              </h3>
            </div>
            
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Soft Dice loss directly optimizes spatial overlap against extreme class imbalance:
            </p>

            <div className="p-3.5 bg-slate-50 dark:bg-[#060a14] rounded-xl border border-slate-200 dark:border-[#16223b] font-mono text-xs text-cyan-700 dark:text-cyan-400 text-center shadow-inner overflow-x-auto">
              <code>L_SoftDice = 1 - [2 * Σ(P_i * Y_i) + ε] / [Σ(P_i²) + Σ(Y_i²) + ε]</code>
            </div>

            <p className="text-[11px] text-slate-500 dark:text-slate-400">
              Total objective balances supervised overlap and unsupervised consistency: L_total = L_sup + λ(t) * L_unsup.
            </p>
          </div>

          {/* Card 4: Semi-Supervised Learning */}
          <div className="p-6 rounded-2xl bg-white dark:bg-[#0c1424] border border-slate-200 dark:border-[#1b2b4d] shadow-md space-y-3.5 hover:border-cyan-500/40 transition-all">
            <div className="flex items-center gap-2">
              <span className="w-2 h-2 rounded-full bg-cyan-500"></span>
              <h3 className="font-bold text-sm sm:text-base text-slate-900 dark:text-slate-100">
                Semi-Supervised Learning (20% Labeled)
              </h3>
            </div>
            
            <p className="text-xs text-slate-600 dark:text-slate-400 leading-relaxed">
              Unlabeled panoramic cases are trained using uncertainty-weighted consistency regularization with synchronized Teacher EMA:
            </p>

            <div className="p-3.5 bg-slate-50 dark:bg-[#060a14] rounded-xl border border-slate-200 dark:border-[#16223b] font-mono text-xs text-cyan-700 dark:text-cyan-400 text-center shadow-inner overflow-x-auto">
              <code>L_unsup = (1/|U|) Σ exp(-σ²(x)) * || f(x; θ) - f(x̃; θ&apos;) ||²</code>
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
