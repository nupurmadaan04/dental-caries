export const PRODUCT_INFO = {
  name: "Dental Caries Clinical AI",
  shortName: "Dental Caries AI",
  subtitle: "Panoramic Radiograph Analysis & Clinical Decision Support",
  version: "2.4.0",
  modality: "Panoramic Dental Radiography (OPG)",
  targetDomain: "Dental Caries Detection, Localization & Pixel-Level Segmentation",
  disclaimer: "FOR CLINICAL DECISION SUPPORT & RESEARCH PURPOSES ONLY. NOT AN AUTONOMOUS DIAGNOSTIC DEVICE. REQUIRES QUALIFIED DENTAL PRACTITIONER VERIFICATION.",
};

export const CLINICAL_STAGES = {
  STAGE_0: {
    key: "STAGE_0",
    label: "Stage 0",
    title: "No Visible Caries Candidates",
    description: "No significant radiolucent candidate areas detected across the dental arch.",
    color: "emerald",
    severity: "Normal / Low Risk",
    recommendation: "Maintain standard routine dental checkups and prophylactic dental hygiene.",
  },
  STAGE_1: {
    key: "STAGE_1",
    label: "Stage 1",
    title: "Early / Initial Caries",
    description: "Localized subtle radiolucency detected (occupying < 1.00% of analyzed radiographic area).",
    color: "cyan",
    severity: "Early Demineralization",
    recommendation: "Recommend clinical visual-tactile examination, remineralization therapy, or targeted bitewing imaging.",
  },
  STAGE_2: {
    key: "STAGE_2",
    label: "Stage 2",
    title: "Moderate Caries",
    description: "Definite localized radiolucent lesion candidates detected (1.00% - 3.00% area).",
    color: "amber",
    severity: "Moderate Enamel/Dentin Caries",
    recommendation: "Recommend targeted clinical dental evaluation, pulp vitality assessment, and restorative consideration.",
  },
  STAGE_3: {
    key: "STAGE_3",
    label: "Stage 3",
    title: "Extensive / Deep Caries",
    description: "Large radiolucent lesion candidate zones approaching or involving deep dentin/pulp (> 3.00% area).",
    color: "rose",
    severity: "Extensive / Deep Caries",
    recommendation: "Immediate clinical intervention and comprehensive endodontic/restorative consultation recommended.",
  },
};

export const CLINICAL_LIMITATIONS = [
  {
    id: 'decision-support',
    title: '1. AI-Assisted Decision Support Nature',
    content: 'The system provides algorithmically highlighted candidate regions of radiolucency to assist clinical review. The output is not an autonomous medical or dental diagnosis.',
  },
  {
    id: 'panoramic-distortion',
    title: '2. Panoramic Radiograph Geometric Distortion',
    content: 'Panoramic radiographs (OPGs) are subject to inherent 15%–30% magnification differences across anterior and posterior arches and slit-beam geometric projection factors.',
  },
  {
    id: 'early-detection',
    title: '3. Early Lesion Detection & Resolution',
    content: 'Incipient enamel demineralization (E1/E2) frequently presents with minimal radiopacity changes on panoramic scans. Supplementary bitewing or periapical radiographs remain the standard of care.',
  },
  {
    id: 'false-positives',
    title: '4. False Positive Considerations',
    content: 'Cervical burnout, Mach band optical phenomena, thin enamel ridges, and anatomical indentations may mimic radiolucency and trigger candidate flags requiring visual dismissal.',
  },
  {
    id: 'false-negatives',
    title: '5. False Negative Considerations',
    content: 'Absence of highlighted candidate zones does not guarantee absence of disease, especially for occlusal pits/fissures and non-cavitated interproximal lesions.',
  },
  {
    id: 'anatomical-overlap',
    title: '6. Anatomical Superimposition & Ghosting',
    content: 'Ghost shadows of the cervical spine, palate, hyoid bone, and overlapping premolar crowns can obscure or simulate lesion margins.',
  },
  {
    id: 'restorations',
    title: '7. Restorations and Metallic Artifacts',
    content: 'Amalgam, metal-ceramic crowns, and radiopaque fillings produce beam-hardening and scattering artifacts that can distort adjacent tooth margins.',
  },
  {
    id: 'image-quality',
    title: '8. Image Quality & Positioning Artifacts',
    content: 'Patient movement, tongue positioning away from the hard palate, or improper head positioning can cause localized blur and alter algorithmic segmentations.',
  },
  {
    id: 'clinical-correlation',
    title: '9. Mandatory Clinical Correlation',
    content: 'All radiographic findings must be correlated with clinical visual-tactile inspection, tooth vitality tests, periodontal probing, and patient dental history.',
  },
  {
    id: 'professional-review',
    title: '10. Qualified Dental Practitioner Requirement',
    content: 'Final radiographic interpretation, clinical staging, and treatment plans must always be performed by a licensed dental professional.',
  },
];

export const CLINICAL_FAQS = [
  {
    id: 'q1',
    question: '1. What is dental caries?',
    answer: 'Dental caries (tooth decay) is a biofilm-mediated, diet-modulated, multifactorial, non-communicable dynamic disease resulting in net mineral loss of dental hard tissues (enamel, dentin, and cementum).',
  },
  {
    id: 'q2',
    question: '2. What does a caries candidate on the radiograph mean?',
    answer: 'A caries candidate indicates an algorithmically identified area of localized radiolucency (lower radiographic density) on the panoramic image that may be consistent with dental demineralization.',
  },
  {
    id: 'q3',
    question: '3. Does a highlighted area definitely mean I have a cavity?',
    answer: 'No. The highlighted zones represent areas for clinical review. Radiographic radiolucency can also be caused by normal anatomical variations, cervical burnout, composite restorations, or overlapping tooth enamel.',
  },
  {
    id: 'q4',
    question: '4. What does Stage 1 mean?',
    answer: 'Stage 1 denotes early or initial caries suspicion, typically occupying less than 1.00% of the radiographic tooth area. These may represent early enamel demineralization that could benefit from remineralization therapy.',
  },
  {
    id: 'q5',
    question: '5. What does Stage 2 mean?',
    answer: 'Stage 2 represents moderate caries candidates (1.00% to 3.00% area) showing clear localized radiolucency that may extend into the outer or middle dentin layer.',
  },
  {
    id: 'q6',
    question: '6. What does Stage 3 mean?',
    answer: 'Stage 3 indicates extensive radiolucency (> 3.00% area) extending deeply toward or into the dental pulp, requiring prompt clinical inspection.',
  },
  {
    id: 'q7',
    question: '7. What does "No caries detected" mean?',
    answer: 'It indicates that the automated screening model did not identify radiographic areas exceeding the detection threshold. It does not rule out incipient, occlusal, or interproximal lesions that are not visible on panoramic imaging.',
  },
  {
    id: 'q8',
    question: '8. Can early caries be missed on a panoramic radiograph?',
    answer: 'Yes. Panoramic radiographs have lower spatial resolution for interproximal enamel surfaces compared to intraoral bitewing radiographs. Bitewing radiographs remain essential for early interproximal caries diagnosis.',
  },
  {
    id: 'q9',
    question: '9. Why does the system sometimes highlight areas that are not cavities?',
    answer: 'Anatomical features such as the dental pulp chamber, cervical burnout (where the tooth narrows at the gumline), and radiolucent filling materials can produce optical contrast similar to demineralization.',
  },
  {
    id: 'q10',
    question: '10. What should I do if an area is flagged?',
    answer: 'The finding should be reviewed by a licensed dentist during an in-person clinical examination, who may confirm tooth vitality, perform visual-tactile assessment, and order supplementary bitewing views if needed.',
  },
  {
    id: 'q11',
    question: '11. Does this system replace a dentist?',
    answer: 'No. The system is strictly an assistive screening and decision-support tool. It cannot perform clinical examinations, determine tooth vitality, diagnose pain, or prescribe dental treatment.',
  },
  {
    id: 'q12',
    question: '12. Why is clinical examination still necessary?',
    answer: 'Radiographs only show hard-tissue mineral density. A physical examination is required to evaluate cavitation, enamel softness, pain symptoms, gingival health, and periodontal status.',
  },
  {
    id: 'q13',
    question: '13. Can dental restorations affect the result?',
    answer: 'Yes. Metallic fillings (amalgam, gold, crowns) absorb X-rays strongly, causing radiopaque shadows and potential edge artifacts, while non-radiopaque resin composites may appear radiolucent.',
  },
  {
    id: 'q14',
    question: '14. Can overlapping teeth affect the analysis?',
    answer: 'Yes. Overcrowded or rotated teeth result in superimposed enamel layers that alter radiographic attenuation, making automated boundary analysis more challenging.',
  },
  {
    id: 'q15',
    question: '15. Why can panoramic radiographs be difficult to interpret?',
    answer: 'Panoramic imaging uses a curved focal trough. Structures outside this layer may appear distorted or blurred, and ghost images from the cervical spine or contralateral jaw can superimpose over tooth roots.',
  },
  {
    id: 'q16',
    question: '16. What does the affected area percentage mean?',
    answer: 'It represents the ratio of segmented positive caries candidate pixels relative to the total analyzed radiographic image area.',
  },
  {
    id: 'q17',
    question: '17. What does lesion confidence mean?',
    answer: 'Confidence reflects the statistical certainty of the deep neural network segmentation output for each detected cluster, ranging from 0% (uncertain) to 100% (high feature agreement).',
  },
  {
    id: 'q18',
    question: '18. What should I do if I disagree with the AI result?',
    answer: 'A clinician can document disagreements directly in the "Clinical Review & Verification" form on the results page. Clinical practitioner judgment always supersedes AI findings.',
  },
  {
    id: 'q19',
    question: '19. Can the same tooth require additional imaging?',
    answer: 'Yes. When panoramic findings are ambiguous, targeted periapical or bitewing radiographs, or Cone Beam CT (CBCT), provide superior local detail.',
  },
  {
    id: 'q20',
    question: '20. Is this system intended for diagnosis or clinical decision support?',
    answer: 'It is designed solely as a clinical decision support and screening aid to help dental clinicians identify candidate regions efficiently during radiographic review.',
  },
];
