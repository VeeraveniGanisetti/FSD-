"""Check the REAL AI model on YOUR photos (needs the packages + downloaded model).

Run from the project folder:

    python tests/real_model_check.py photo1.jpg photo2.png
    python tests/real_model_check.py --folder sample_images

For --folder, make sub-folders named after the model's classes, e.g.
    sample_images/Plastic/bottle1.jpg
    sample_images/Glass/jar.jpg
The script prints the prediction for every image and an overall accuracy.
Classes: Battery, Biological, Cardboard, Clothes, Glass, Metal, Paper, Plastic, Shoes, Trash
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import guidance  # noqa: E402
from classifier import EXPECTED_LABELS, ClassifierUnavailable, WasteClassifier  # noqa: E402
from imaging import ImageRejected, load_image  # noqa: E402

EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")


def collect(arguments):
    """Return a list of (path, true_label_or_None)."""
    if arguments and arguments[0] == "--folder":
        root = arguments[1] if len(arguments) > 1 else "sample_images"
        found = []
        for label in sorted(os.listdir(root)):
            folder = os.path.join(root, label)
            if os.path.isdir(folder):
                if label not in EXPECTED_LABELS:
                    print(f"Skipping folder {label!r}: not one of {', '.join(EXPECTED_LABELS)}")
                    continue
                found += [(os.path.join(folder, n), label) for n in sorted(os.listdir(folder))
                          if n.lower().endswith(EXTENSIONS)]
        return found
    return [(path, None) for path in arguments]


def main():
    items = collect(sys.argv[1:])
    if not items:
        print(__doc__)
        return 2
    model = WasteClassifier()
    try:
        model.load()
    except ClassifierUnavailable as error:
        print("Model not available: " + str(error))
        return 1
    right = total = 0
    for path, truth in items:
        try:
            predictions = model.predict(load_image(open(path, "rb").read()), top_k=3)
        except (ImageRejected, OSError) as error:
            print(f"{path}: SKIPPED ({error})")
            continue
        result = guidance.interpret(predictions)
        line = f"{path}: {predictions[0]['label']} {predictions[0]['probability']:.0%}  [{result['status']}]"
        if truth:
            total += 1
            right += predictions[0]["label"] == truth
            line += f"   expected {truth}: {'OK' if predictions[0]['label'] == truth else 'WRONG'}"
        print(line)
    if total:
        print(f"\nAccuracy on your photos: {right}/{total} = {right / total:.0%}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
