"""Waste image classifier.

This runs a REAL pretrained waste-classification model inside the Flask server:

    prithivMLmods/Augmented-Waste-Classifier-SigLIP2   (Hugging Face, Apache-2.0)

It is a SigLIP2-base model fine-tuned to sort photos into 10 classes.
The model is downloaded once (about 372 MB) into the "models" folder next to
this file, then reused offline.

If the model cannot load or run, this file raises an error. It NEVER invents a
prediction.
"""
import os
import threading

MODEL_ID = "prithivMLmods/Augmented-Waste-Classifier-SigLIP2"
# Pinned to an exact version of the model so results do not change unexpectedly.
MODEL_REVISION = "d203b52311f70ac3245d5fb8345d88344bec138e"
MODEL_LICENSE = "Apache-2.0"

# The classes listed on the model card / config.json. After loading, we check the
# real model reports exactly these, so the guidance mapping can be trusted.
EXPECTED_LABELS = (
    "Battery", "Biological", "Cardboard", "Clothes", "Glass",
    "Metal", "Paper", "Plastic", "Shoes", "Trash",
)

DEFAULT_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")

STATE_NOT_LOADED = "not_loaded"
STATE_LOADING = "loading"
STATE_READY = "ready"
STATE_ERROR = "error"


class ClassifierUnavailable(Exception):
    """The model is not loaded (still loading, not downloaded, or libraries missing)."""


class InferenceError(Exception):
    """The model was loaded but failed while analysing an image."""


class WasteClassifier:
    def __init__(self, model_id=MODEL_ID, revision=MODEL_REVISION, cache_dir=DEFAULT_CACHE_DIR):
        self.model_id = model_id
        self.revision = revision
        self.cache_dir = cache_dir
        self._lock = threading.Lock()          # protects loading and prediction
        self._state = STATE_NOT_LOADED
        self._message = "The AI model has not been loaded yet."
        self._model = None
        self._processor = None
        self._labels = ()

    # ------------------------------------------------------------------ status
    def status(self):
        return {
            "state": self._state,
            "message": self._message,
            "model_id": self.model_id,
            "revision": self.revision,
            "license": MODEL_LICENSE,
            "labels": list(self._labels or EXPECTED_LABELS),
        }

    @property
    def is_ready(self):
        return self._state == STATE_READY

    # ----------------------------------------------------------------- loading
    def load(self):
        """Download (first run only) and load the model. Safe to call repeatedly."""
        if self._state == STATE_READY:
            return
        if not self._lock.acquire(blocking=False):
            raise ClassifierUnavailable("The AI model is still loading. Please wait a moment and try again.")
        try:
            if self._state == STATE_READY:
                return
            self._state = STATE_LOADING
            self._message = "Loading the AI model. The first run downloads about 372 MB, so this can take a few minutes."
            try:
                import torch  # noqa: F401  (imported here so the website still starts without it)
                from transformers import AutoImageProcessor, AutoModelForImageClassification
            except ImportError as error:
                self._fail("The AI libraries (torch, transformers) are not installed. "
                           "Run: python -m pip install -r requirements.txt")
                raise ClassifierUnavailable(self._message) from error

            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                processor = AutoImageProcessor.from_pretrained(
                    self.model_id, revision=self.revision, cache_dir=self.cache_dir)
                model = AutoModelForImageClassification.from_pretrained(
                    self.model_id, revision=self.revision, cache_dir=self.cache_dir,
                    use_safetensors=True)
                model.eval()
            except Exception as error:  # download / disk / corrupt-cache problems
                self._fail("The AI model could not be loaded. Check your internet connection for the "
                           "first-time download, then restart. Details: " + _short(error))
                raise ClassifierUnavailable(self._message) from error

            labels = tuple(model.config.id2label[i] for i in sorted(model.config.id2label))
            if set(labels) != set(EXPECTED_LABELS):
                self._fail("The downloaded model reports unexpected classes: " + ", ".join(labels)
                           + ". Refusing to use it.")
                raise ClassifierUnavailable(self._message)

            self._processor, self._model, self._labels = processor, model, labels
            self._state = STATE_READY
            self._message = "AI model ready."
        finally:
            self._lock.release()

    def _fail(self, message):
        self._state = STATE_ERROR
        self._message = message

    def warm_up_async(self):
        """Start loading in the background so the first scan is not slow."""
        def work():
            try:
                self.load()
            except ClassifierUnavailable:
                pass  # the status already explains what happened
        threading.Thread(target=work, name="model-warmup", daemon=True).start()

    # -------------------------------------------------------------- prediction
    def predict(self, image, top_k=3):
        """Classify a PIL RGB image. Returns [{"label": str, "probability": float}, ...] best first."""
        if self._state != STATE_READY:
            self.load()   # tries to load; raises ClassifierUnavailable with a clear message
        import torch
        try:
            with self._lock:
                with torch.inference_mode():
                    inputs = self._processor(images=image, return_tensors="pt")
                    logits = self._model(**inputs).logits[0]
                    probabilities = torch.softmax(logits, dim=-1).tolist()
        except Exception as error:
            raise InferenceError("The model failed while analysing the image: " + _short(error)) from error

        if len(probabilities) != len(self._labels):
            raise InferenceError("The model returned an unexpected number of scores.")
        total = sum(probabilities)
        if any(p != p for p in probabilities) or abs(total - 1.0) > 1e-3:   # NaN / not a distribution
            raise InferenceError("The model returned invalid scores.")

        ranked = sorted(zip(self._labels, probabilities), key=lambda pair: pair[1], reverse=True)
        return [{"label": name, "probability": float(prob)} for name, prob in ranked[:top_k]]


def _short(error):
    text = f"{type(error).__name__}: {error}"
    return text if len(text) <= 300 else text[:297] + "..."
