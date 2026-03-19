NeuroScan AI 🧠

> Multimodal Early Alzheimer's Detection Platform using Deep Learning

## Overview
NeuroScan AI is a clinical decision support system that detects Alzheimer's Disease at the earliest possible stage by fusing MRI scans, Amyloid PET scans, and blood biomarkers into a unified AI pipeline.

## What Makes It Different
Every existing tool uses a single modality. NeuroScan AI fuses three:

| Modality | What It Detects | Stage |
|----------|----------------|-------|
| Blood Biomarkers (p-tau, Aβ42/40) | Amyloid buildup | Preclinical |
| Amyloid PET | Plaque deposits | Preclinical → EMCI |
| MRI (T1 MPRAGE) | Brain atrophy | EMCI → AD |

## Architecture
```
MRI Scan → EfficientNetB4 CNN ──┐
Amyloid PET → ResNet50 CNN ─────┼→ Fusion Layer → CN/Preclinical/EMCI/LMCI/AD
Blood Tests → XGBoost ──────────┘
                    ↓
             Grad-CAM Heatmap + PDF Report
```

## Current Status
### Phase 1 — MRI Model (In Progress)
- [x] Environment setup (TensorFlow + CUDA on RTX 3050)
- [x] ADNI dataset access approved
- [x] DICOM to NIfTI conversion pipeline
- [x] EfficientNetB4 binary classifier (AD vs CN)
- [x] Patient-level train/test split (no data leakage)
- [ ] More unique patients (target: 300+ per class)
- [ ] 5-class classifier (CN/SMC/EMCI/LMCI/AD)

### Phase 2 — Amyloid PET Model (Planned)
- [ ] Download Amyloid PET scans from ADNI
- [ ] ResNet50 PET classifier
- [ ] OASIS-3 dataset integration

### Phase 3 — Blood Biomarker Model (Planned)
- [ ] ADNIMERGE.csv feature engineering
- [ ] XGBoost on p-tau, Aβ42/40, NfL, ApoE4
- [ ] Preclinical stage detection

### Phase 4 — Fusion + Clinical Output (Planned)
- [ ] Late fusion layer combining all three models
- [ ] Grad-CAM explainability heatmaps
- [ ] PDF clinical report generation
- [ ] Flask REST API
- [ ] React + Tailwind frontend

## Dataset
- **ADNI** (Alzheimer's Disease Neuroimaging Initiative) — approved access
- **OASIS-3** — pending registration
- **Kaggle** Alzheimer's MRI datasets

## Model Performance (v0.2)
| Metric | Score |
|--------|-------|
| Test Accuracy | 42% (limited by small unique patient count) |
| Test AUC | 0.64 |
| Training Data | 137 unique patients (65 AD + 72 CN) |
| Architecture | EfficientNetB4 + Dense layers |

## Tech Stack
| Component | Technology |
|-----------|-----------|
| MRI Model | EfficientNetB4 (TensorFlow/Keras) |
| PET Model | ResNet50 (planned) |
| Blood Model | XGBoost (planned) |
| Explainability | Grad-CAM |
| Preprocessing | dcm2niix, nibabel, nilearn |
| Backend | Flask (planned) |
| Frontend | React + Tailwind (planned) |
| Database | PostgreSQL + MinIO (planned) |

## Team
Built by a 6-member team from Sri Krishna College of Engineering and Technology, Coimbatore. Members are Antonius Jairus, Dhanya Shri, Sweatha, Sharukesh, Noorul, Hariharasudhan.

## License
Research use only. ADNI data subject to ADNI data use agreement.
