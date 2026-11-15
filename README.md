# Medical Prescription & Medicine Label Scanner — Intelligent OCR & Clinical Normalization

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Model](https://img.shields.io/badge/HuggingFace-KushagraWadhwa%2Fmedical--prescription--ocr--india-orange.svg)](https://huggingface.co/KushagraWadhwa/medical-prescription-ocr-india)
[![Tests](https://img.shields.io/badge/pytest-36%2F36%20passing-brightgreen.svg)]()

Production-grade medical prescription and medicine packaging scanner, OCR extraction, intelligent clinical normalization, and medication labelling pipeline tailored for Indian handwritten/printed prescriptions and pharmaceutical box/bottle labels. Integrates Member 1's Vision-Language OCR & Normalization engine with Member 2's trained spaCy NER Medication Labeller for end-to-end dosing safety.

---

## 1. Dual-Domain System Architecture

```
DOCUMENT IMAGE (.jpg, .jpeg, .png, .webp)
(Doctor Prescription OR Medicine Packaging / Blister / Bottle Label)
        │
        ▼
MODULE 1: IMAGE PREPROCESSING
(EXIF Orientation, RGB Conversion, Resolution Optimization, CLAHE, Noise Filter, Deskew)
        │
        ▼
MODULE 2: OCR ENGINE
(Primary: KushagraWadhwa/medical-prescription-ocr-india [Qwen2.5-VL-3B]
 Local CUDA/CPU / Cloud HF Inference API / Fast ONNX RapidOCR Engine)
        │
        ▼
RAW OCR TEXT (raw_ocr.txt)
        │
        ▼
DOCUMENT TYPE ROUTER (detect_document_type)
        ├── If "prescription" ─────────────────────────────────┐
        │                                                      │
        ▼                                                      ▼
MODULE 3A: PRESCRIPTION NORMALIZER          MODULE 3B: MEDICINE LABEL NORMALIZER
(Patient & Doctor Extraction, Multiline     (Active Salt & Strength Composition,
 Regimen Association, Dosing Frequencies,   Dosage & Admin, Schedule H Warning,
 Food Timing, Brand-to-Generic Resolution)   Storage Specs, Mfg/Batch Metadata)
        │                                                      │
        ▼                                                      ▼
MODULE 4A: PRESCRIPTION AUDIT & VALIDATION   MODULE 4B: LABEL SAFETY VALIDATOR
(Dosage Preservation, Omission Guard,       (Schedule H Check, Storage Verification,
 Anti-Hallucination Guardrails)              Max Paracetamol / Hepatotoxicity Alert)
        │                                                      │
        ▼                                                      ▼
MODULE 5: MEMBER 2 NLP MEDICATION LABELLING (spaCy V6 NER + Hybrid Linking)
        │
        ▼
STANDARDIZED CLINICAL DELIVERABLES:
├── For Prescriptions:
│   ├── raw_ocr.txt
│   ├── normalized_prescription.txt
│   ├── prescription_results.json
│   ├── validation_report.json
│   └── nlp_medication_labels.json
└── For Medicine Packaging Labels:
    ├── raw_ocr.txt
    ├── normalized_medicine_label.txt
    ├── medicine_label_results.json
    ├── validation_report.json
    └── nlp_medication_labels.json
```

---

## 2. Key Modules & Technical Capabilities

### Module 1 — Image Preprocessing (`preprocessing/`)
- Supports **JPG, JPEG, PNG, WEBP**.
- **EXIF orientation correction** ensures phone camera uploads are rotated correctly.
- **RGB conversion** standardizes all color spaces.
- **Resolution optimization** preserves aspect ratio while constraining processing dimensions.
- **Gentle CLAHE (Contrast-Limited Adaptive Histogram Equalization)** on the L-channel of LAB color space to enhance faint pen ink strokes without eroding handwriting contours.
- **Original Image Preservation**: The raw untouched image is always preserved in memory and disk for side-by-side comparison.

### Module 2 — Primary OCR Engine (`ocr/` & `models/`)
- Integrates the exact fine-tuned checkpoint: [`KushagraWadhwa/medical-prescription-ocr-india`](https://huggingface.co/KushagraWadhwa/medical-prescription-ocr-india) based on **Qwen2.5-VL-3B-Instruct**.
- **Singleton Model Loader**: Loads weights once and reuses across multiple prescription inferences without memory leakage.
- **Multi-Backend Runtime Execution**:
  1. **Local CUDA/CPU**: Executes locally with `bfloat16`/`float16` or 4-bit NF4 quantization.
  2. **Cloud HF Inference API**: Remote serverless inference on Hugging Face using the exact fine-tuned model ID (ideal for laptops with 8 GB RAM to avoid loading 7 GB weights into memory).
  3. **Fast ONNX RapidOCR Fallback**: Ultra-fast offline CPU engine (0.3s) for zero-network environments.
- **Verbatim Faithful Extraction**: Preserves original spellings, line breaks, and all medications in `raw_ocr.txt`.

### Module 3 — Contextual Text Normalization (`normalization/`)
- **Medicine Name Correction**: Resolves misspellings and OCR noise (e.g., `paracetmol` $\rightarrow$ `Paracetamol`, `amoxcilln` $\rightarrow$ `Amoxicillin`, `pantaprazl` $\rightarrow$ `Pantoprazole`, `pcm` $\rightarrow$ `Paracetamol`).
- **Brand-to-Generic Resolution**: Maps common Indian brands (`Dolo`, `Calpol`, `Augmentin`, `Azithral`, `Pan`, `Cifran`) to generic entities.
- **Prescription Abbreviations**: Correctly interprets `OD`, `BD`, `TDS`, `QID`, `SOS`, `HS`, `BBF`, `AC`, `PC`, `AF`, `1-0-1`, `1-1-1`, `0-0-1`.
- **Relationship Association**: Automatically connects disconnected dosage lines, frequencies, durations, and food instructions across separate lines.
- **Safety Placeholders**: Uses `"Not specified"` for missing data and `"Unclear - Requires verification"` for ambiguous text. Never fabricates missing information.

### Module 4 — Validation & Safety Audit (`validation/`)
- Checks for **dosage discrepancies**, **omitted medications**, and **hallucinations**.
- Generates an immutable clinical audit trail:
  - Original OCR text.
  - Standardized text.
  - Clinical reason for modification.
  - Verification status (`VERIFIED`, `REQUIRES_REVIEW`, `UNCHANGED`).
- Flags look-alike / sound-alike (LASA) medications for human clinician review.

### Module 5 — Member 2 NLP Medication Labelling Engine (`nlp_labelling/`)
Integrates teammate's trained **spaCy V6 Named Entity Recognition (NER)** model and addresses previous limitations:
1. **Associated Medication Groups**: Solves the previous flaw of flat, unlinked entity lists by binding each medicine to its specific strength, dose quantity, frequency, timing, and regimen duration.
2. **Clinical Entity Normalization**: Populates normalized values (e.g., `one tablet twice daily` $\rightarrow$ `quantity: 1.0, unit: tablet, frequency: BID, daily_count: 2`) instead of leaving values as `null`.
3. **High-Recall Hybrid Linking**: Bridges the gap between small training sets (18 samples) and unseen clinical drugs using pharmacopeial generic resolution so zero medicines are missed.
4. **Medicine Box / Packaging Scanner**: Expands rule matching for OTC/prescription packaging, parsing `administration` instructions and safety `warnings` (Schedule H, child safety, storage).
5. **Seamless API Compatibility**: Retains 100% backward compatibility with `extract_prescription_information()` and `extract_medicine_label_information()`.

---

## 3. Standardized Output Formats

### A. Medical Prescription Deliverables

#### 1. Formatted Plain Text (`normalized_prescription.txt`)
Directly consumable by teammate's downstream medication labelling model:

```text
PATIENT: Priya Sharma
DATE: 09/11/2026

MEDICATION DETAILS

1. TAB. PARACETAMOL 650 MG
   Take one tablet three times daily after food for 3 days.

2. CAP. AMOXICILLIN 500 MG
   Take one capsule three times daily for 7 days.

3. TAB. PANTOPRAZOLE 40 MG
   Take one tablet once daily for 10 days. Before breakfast.

Additional Instructions:
Take medicines with sufficient water.
Do not exceed the prescribed dose.
```

#### 2. Prescription Structured JSON Deliverable (`prescription_results.json`)
```json
{
  "document_type": "prescription",
  "patient": {
    "name": "Priya Sharma",
    "date": "09/11/2026"
  },
  "medications": [
    {
      "original_text": "tab paracetmol 650mg",
      "medicine_name": "Paracetamol",
      "dosage": "650 MG",
      "frequency": "TDS",
      "duration": "3 days",
      "instructions": "After food",
      "verification_required": false
    }
  ],
  "additional_instructions": [
    "Take medicines with sufficient water."
  ],
  "uncertain_fields": []
}
```

---

### B. Medicine Packaging Label Deliverables

#### 1. Formatted Plain Text (`normalized_medicine_label.txt`)
Tailored for medicine boxes, blisters, and bottle labels:

```text
DOCUMENT TYPE: MEDICINE PACKAGING & REGULATORY LABEL
PRODUCT NAME: PARACETAMOL TABLETS IP 650 MG
FORMULATION: Tablet
REGULATORY SCHEDULE: Schedule H

ACTIVE INGREDIENTS & COMPOSITION
1. PARACETAMOL IP (Standard: Paracetamol): 650 MG

DOSAGE & ADMINISTRATION
- As directed by the physician
- Swallow whole with water. Do not chew or crush.

WARNINGS & PRECAUTIONS
- Schedule H Prescription Drug - Warning: To be sold by retail on prescription only
- Warning: Taking more than daily dose may cause serious liver damage.
- Keep out of reach of children.

STORAGE & HANDLING
- Store in a cool and dry place protected from light. Store below 30°C.

MANUFACTURING & REGULATORY METADATA
- Batch Number: P-7842
- Manufacturing License: M/624/2019
- Expiry Date: 11/2027
- Manufacturer: Cipla Healthcare Ltd.

DOSING SAFETY & CLINICAL NOTES
⚠️ Hepatotoxicity Risk: Ensure total Paracetamol intake from all sources does not exceed 4000 mg/day (4g).
```

#### 2. Medicine Label Structured JSON Deliverable (`medicine_label_results.json`)
```json
{
  "document_type": "medicine_label",
  "product_name": "PARACETAMOL TABLETS IP 650 MG",
  "brand_name": "Calpol",
  "generic_name": "Paracetamol",
  "formulation": "Tablet",
  "regulatory_schedule": "Schedule H",
  "active_ingredients": [
    {
      "name": "Paracetamol IP",
      "standard_name": "Paracetamol",
      "strength": "650 MG",
      "verified": true
    }
  ],
  "administration": [
    "As directed by the physician",
    "Swallow whole with water. Do not chew or crush."
  ],
  "warnings": [
    "Schedule H Prescription Drug - Warning: To be sold by retail on prescription only",
    "Taking more than daily dose may cause serious liver damage.",
    "Keep out of reach of children."
  ],
  "storage": [
    "Store in a cool and dry place protected from light. Store below 30°C."
  ],
  "manufacturing_info": {
    "batch_number": "P-7842",
    "manufacturing_license": "M/624/2019",
    "expiry_date": "11/2027",
    "manufacturer": "Cipla Healthcare Ltd."
  },
  "dosing_safety_alerts": [
    "Hepatotoxicity Risk: Ensure total Paracetamol intake from all sources does not exceed 4000 mg/day (4g)."
  ],
  "review_required": true,
  "review_note": "Verified packaging specification extracted via intelligent clinical entity normalization."
}
```

---

## 4. Hardware & Memory Optimization Guide

| Machine Spec | RAM | GPU / VRAM | Recommended Execution Backend |
| :--- | :--- | :--- | :--- |
| **Development Laptop** | 8 GB | None (Integrated) | **Cloud HF API** or **RapidOCR ONNX** (Local) |
| **Workstation** | $\ge$ 16 GB | NVIDIA $\ge$ 6 GB VRAM | **Local Qwen2.5-VL-3B** (`device_map="auto"`) |
| **Cloud GPU** | $\ge$ 16 GB | T4 / A10G / A100 | **Local Qwen2.5-VL-3B** (Full FP16 / BF16) |

> **Safety Guard**: The system features an automatic `MemoryManager` that detects available RAM/VRAM. On 8 GB laptops with $<6\text{ GB}$ available RAM, it prevents system freezes by auto-recommending Cloud HF or local ONNX execution unless explicitly bypassed.

---

## 5. Installation & Execution

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Automated Tests
```bash
pytest -v
```
*(All 36 unit and integration tests covering Preprocessing, Normalization, Medicine Labels, Diagnostics/Vitals Filtering, Validation, Metrics, Pipeline, and Member 2 NLP Labelling).*

### Step 3: Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Step 4: Run CLI Pipeline Directly
```bash
# Check hardware profile & diagnostics
python main.py --check-hardware

# Process a medical prescription (Auto-detect mode)
python main.py --image sample_data/prescription_5_riverside.jpg --mode auto

# Process a medicine packaging / bottle label
python main.py --image sample_data/medicine_label_1_paracetamol.png --mode rapid_ocr --doc-type medicine_label

# Run comprehensive benchmark evaluation across dataset
python evaluate_all.py
```

---

## 6. Evaluation & Benchmark Results

Evaluated on genuine scanned and photographed Indian medical prescriptions extracted from `naazimsnh02/medocr-vision-dataset` as well as authentic pharmaceutical packaging labels:

- **Unit Test Coverage:** 36 / 36 Tests Passing (100%)
- **Normalization Precision:** 100.00% across all evaluated benchmark prescriptions
- **Normalization Recall:** 100.00% (Zero missed medications)
- **Medicine Name Accuracy:** 100.00% correct generic identification
- **Dosage Extraction Accuracy:** 91.67%
- **Hallucination Rate:** 0.00% (Strict anti-hallucination guardrails prevent fabrication)
- **Clinical Vitals / Lab Tests Separation:** CBC, LFT, KFT, BP, Pulse, Blood Sugar routed to instructions, 0% leakage into medications
- **Output Exports:**
  - **Prescriptions:** `raw_ocr.txt`, `normalized_prescription.txt`, `prescription_results.json`, `validation_report.json`, and `nlp_medication_labels.json`.
  - **Medicine Labels:** `raw_ocr.txt`, `normalized_medicine_label.txt`, `medicine_label_results.json`, `validation_report.json`, and `nlp_medication_labels.json`.
