"""Download and test-load the waste model once, before starting the website.

    python setup_model.py

Needs internet for the first run only (about 372 MB). Afterwards the model is
stored in the "models" folder and ECHO SCAN works offline.
Exit code 0 = model downloaded and loaded OK, 1 = something went wrong.
"""
import os
import sys

from classifier import DEFAULT_CACHE_DIR, MODEL_ID, MODEL_REVISION, WasteClassifier


def main():
    print(f"Model:    {MODEL_ID}")
    print(f"Revision: {MODEL_REVISION}")
    print("Downloading / loading (first run is about 372 MB)...")
    classifier = WasteClassifier()
    try:
        classifier.load()
    except Exception as error:   # ClassifierUnavailable or anything unexpected
        print("\nFAILED: " + str(error))
        return 1
    print("OK: model loaded. Classes: " + ", ".join(classifier.status()["labels"]))
    with open(os.path.join(DEFAULT_CACHE_DIR, ".ready"), "w") as marker:   # lets run_windows.bat skip this step next time
        marker.write(MODEL_REVISION)
    return 0


if __name__ == "__main__":
    sys.exit(main())
