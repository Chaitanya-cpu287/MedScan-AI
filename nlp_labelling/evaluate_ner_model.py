import json
import spacy

from spacy.training import Example
from spacy.training import offsets_to_biluo_tags


MODEL_PATH = "trained_model_v6"
TEST_FILE = "six_label_validation_data.json"


def load_test_data(filename):

    with open(filename, "r", encoding="utf-8") as file:
        return json.load(file)


def create_examples(nlp, records):

    examples = []

    for record in records:

        text = record["text"]
        entities = record["entities"]

        doc = nlp.make_doc(text)

        tags = offsets_to_biluo_tags(
            doc,
            entities
        )

        if "-" in tags:
            raise ValueError(
                f"Entity alignment error: {text}"
            )

        example = Example.from_dict(
            doc,
            {"entities": entities}
        )

        examples.append(example)

    return examples


def main():

    print("Loading V6 model...")

    nlp = spacy.load(MODEL_PATH)

    test_data = load_test_data(TEST_FILE)

    test_examples = create_examples(
        nlp,
        test_data
    )

    print("Test records:", len(test_examples))

    scores = nlp.evaluate(test_examples)

    print("\n========== V6 OVERALL EVALUATION ==========")

    print(
        f"Precision: {scores['ents_p']:.4f} "
        f"({scores['ents_p'] * 100:.2f}%)"
    )

    print(
        f"Recall: {scores['ents_r']:.4f} "
        f"({scores['ents_r'] * 100:.2f}%)"
    )

    print(
        f"F1-score: {scores['ents_f']:.4f} "
        f"({scores['ents_f'] * 100:.2f}%)"
    )

    print("\n========== LABEL-WISE EVALUATION ==========")

    label_scores = scores.get("ents_per_type", {})

    labels = [
        "MEDICINE",
        "STRENGTH",
        "DOSE",
        "FREQUENCY",
        "TIMING",
        "DURATION",
        "ADMINISTRATION",
        "WARNING"
    ]

    for label in labels:

        metrics = label_scores.get(label, {})

        precision = metrics.get("p", 0.0)
        recall = metrics.get("r", 0.0)
        f1 = metrics.get("f", 0.0)

        print(f"\n{label}")

        print(f"Precision: {precision * 100:.2f}%")
        print(f"Recall: {recall * 100:.2f}%")
        print(f"F1-score: {f1 * 100:.2f}%")

    print("\n========== EVALUATION COMPLETED ==========")


if __name__ == "__main__":
    main()