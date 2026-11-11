from structured_output import (
    extract_prescription_information,
    extract_medicine_label_information
)

import json


# TEST 1: Simple prescription
prescription_1 = """
Prescription

Patient: Rahul

Medicine: Azithromycin 500 mg

Take one tablet once daily for 3 days.
"""


# TEST 2: Prescription with timing
prescription_2 = """
Prescription

Medicine: Metformin 500 mg

Take one tablet twice daily after meals for 30 days.
"""


# TEST 3: Multiple medicines
prescription_3 = """
Prescription

1. Paracetamol 650 mg
   Take one tablet every 8 hours for 3 days.

2. Cetirizine 10 mg
   Take one tablet at bedtime for 5 days.
"""


# TEST 4: Medicine label
label_1 = """
CETIRIZINE TABLETS IP

Each tablet contains:
Cetirizine Hydrochloride IP 10 mg.

Dosage: As directed by the physician.
"""


# TEST 5: Medicine label with warnings
label_2 = """
PARACETAMOL TABLETS IP 650 mg

Each tablet contains:
Paracetamol IP 650 mg.

Swallow with water.
Do not exceed the recommended dosage.
Keep out of reach of children.
"""


# Run prescription tests
prescription_tests = {
    "Test 1": prescription_1,
    "Test 2": prescription_2,
    "Test 3": prescription_3
}

for test_name, text in prescription_tests.items():

    print("\n" + "=" * 60)
    print(test_name + " - PRESCRIPTION")
    print("=" * 60)

    result = extract_prescription_information(text)

    print(json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ))


# Run medicine label tests
label_tests = {
    "Test 4": label_1,
    "Test 5": label_2
}

for test_name, text in label_tests.items():

    print("\n" + "=" * 60)
    print(test_name + " - MEDICINE LABEL")
    print("=" * 60)

    result = extract_medicine_label_information(text)

    print(json.dumps(
        result,
        indent=4,
        ensure_ascii=False
    ))