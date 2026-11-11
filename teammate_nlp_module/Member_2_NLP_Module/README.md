# Member 2 – NLP and Machine Learning Module

## Project
AI-Based Prescription and Medicine Label Scanner for Dosing and Medication Safety.

## Module Description
This module uses a trained spaCy Named Entity Recognition (NER) model to extract medication-related information from OCR text.

## Extracted Information

### Prescription
- Medicine name
- Strength
- Dose
- Frequency
- Timing
- Duration

### Medicine Label
- Medicine name
- Strength
- Administration instructions
- Warnings

## Project Files

- `structured_output.py` – Main extraction module.
- `trained_model_v6/` – Trained spaCy NER model.
- `prescription_structured_output.json` – Sample prescription output.
- `medicine_label_structured_output.json` – Sample medicine label output.
- `requirements.txt` – Required Python dependency.

## Installation

Install the dependency:

```bash
pip install -r requirements.txt
```

## Integration with Member 3

Import the extraction functions:

```python
from structured_output import (
    extract_prescription_information,
    extract_medicine_label_information
)
```

For prescription OCR text:

```python
prescription_result = extract_prescription_information(ocr_text)
```

For medicine label OCR text:

```python
label_result = extract_medicine_label_information(ocr_text)
```

The `ocr_text` argument should contain the text returned by Member 1's OCR module.

Both functions return Python dictionaries that can be converted into JSON.

## Model Evaluation

The V6 model achieved the following results on the 18-record validation dataset:

- Precision: 90.91%
- Recall: 83.33%
- F1-score: 86.96%

These are preliminary results from a small validation dataset. The validation data was also used during model selection, so these scores are not an independent final test.

## Important Limitations

- OCR errors can affect extraction.
- Some medication instructions may not be detected.
- Empty fields mean that the system did not extract a value; they do not prove the information is absent.
- Extracted information must be checked against the original prescription or medicine label.
- The system is a screening aid, not a substitute for professional medical judgment.
- It must not independently recommend or modify medication doses.