"""Automated tests for ECHO SCAN.   Run from the project folder with:

    python -m unittest discover -s tests -v

IMPORTANT: these tests check the website, database, validation and guidance.
They use a FAKE classifier (defined below, test-only) so they can run in seconds
without downloading the 372 MB model. They do NOT prove the AI model's accuracy.
Real-model inference is checked separately by tests/real_model_check.py.
"""
import io
import os
import re
import shutil
import sqlite3
import sys
import tempfile
import unittest
from importlib.util import find_spec

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PIL import Image  # noqa: E402

import classifier as classifier_module  # noqa: E402
import guidance  # noqa: E402
from app import BASE_DIR, create_app  # noqa: E402
from classifier import ClassifierUnavailable, InferenceError, WasteClassifier  # noqa: E402


class FakeClassifier:
    """TEST ONLY. Returns whatever the test tells it to. Never used by the real app."""
    model_id = "fake/test-model"

    def __init__(self, predictions=None, error=None):
        self.predictions = predictions or [
            {"label": "Plastic", "probability": 0.93},
            {"label": "Glass", "probability": 0.04},
            {"label": "Metal", "probability": 0.02},
        ]
        self.error = error
        self.calls = 0

    def status(self):
        return {"state": "ready", "message": "fake", "model_id": self.model_id, "labels": []}

    def predict(self, image, top_k=3):
        self.calls += 1
        assert image.mode == "RGB"
        if self.error:
            raise self.error
        return self.predictions[:top_k]


def make_image(fmt="PNG", size=(64, 48), color=(30, 160, 90)):
    buffer = io.BytesIO()
    Image.new("RGB", size, color).save(buffer, fmt)
    return buffer.getvalue()


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.db_path = os.path.join(self.tmp, "instance", "test.db")
        self.fake = FakeClassifier()
        self.app = create_app({"TESTING": True, "DATABASE": self.db_path}, classifier=self.fake)
        self.client = self.app.test_client()

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def scan(self, data=None, filename="photo.png", mimetype="image/png", extra=None):
        data = make_image() if data is None else data
        form = {"image": (io.BytesIO(data), filename, mimetype)}
        form.update(extra or {})
        return self.client.post("/api/classify", data=form, content_type="multipart/form-data")


class TestPagesAndAuthRemoval(AppTestCase):
    def test_home_loads_without_account(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        html = response.get_data(as_text=True)
        self.assertIn("ECHO", html)
        for word in ("login", "log in", "logout", "sign in", "register", "sign up", "password"):
            self.assertNotIn(word, html.lower(), f"home page still mentions {word!r}")

    def test_auth_routes_do_not_exist(self):
        for path in ("/login", "/register", "/logout", "/signup", "/signin", "/auth/login"):
            for method in ("get", "post"):
                self.assertEqual(getattr(self.client, method)(path).status_code, 404, f"{method} {path}")

    def test_no_auth_code_left_in_sources(self):
        pattern = re.compile(r"login|logout|register|session\[|flask_login|password", re.I)
        for name in ("app.py", "database.py", "guidance.py", "classifier.py", "imaging.py",
                     "templates/index.html", "static/js/app.js", "static/css/style.css"):
            with open(os.path.join(BASE_DIR, name), encoding="utf-8") as handle:
                for number, line in enumerate(handle, 1):
                    self.assertIsNone(pattern.search(line), f"{name}:{number} mentions auth: {line.strip()[:80]}")

    def test_security_headers(self):
        response = self.client.get("/")
        self.assertEqual(response.headers["X-Content-Type-Options"], "nosniff")
        self.assertIn("script-src 'self'", response.headers["Content-Security-Policy"])

    def test_unknown_api_path_is_json_404(self):
        response = self.client.get("/api/nothing")
        self.assertEqual(response.status_code, 404)
        self.assertIn("error", response.get_json())

    def test_health_reports_model_status(self):
        data = self.client.get("/api/health").get_json()
        self.assertTrue(data["ok"])
        self.assertEqual(data["model"]["state"], "ready")


class TestFrontendReferences(AppTestCase):
    def setUp(self):
        super().setUp()
        with open(os.path.join(BASE_DIR, "templates", "index.html"), encoding="utf-8") as handle:
            self.html = handle.read()
        with open(os.path.join(BASE_DIR, "static", "js", "app.js"), encoding="utf-8") as handle:
            self.js = handle.read()

    def test_every_element_id_used_by_javascript_exists_in_html(self):
        html_ids = set(re.findall(r'id="([^"]+)"', self.html))
        js_ids = set(re.findall(r'\$\("([^"]+)"\)', self.js))
        self.assertTrue(js_ids, "no element ids found in app.js")
        self.assertEqual(sorted(js_ids - html_ids), [], "app.js uses ids missing from index.html")

    def test_every_api_url_used_by_javascript_is_a_real_route(self):
        routes = {rule.rule for rule in self.app.url_map.iter_rules()}
        urls = set(re.findall(r'"(/api/[a-z/]+)"', self.js))
        self.assertTrue(urls)
        self.assertTrue(urls <= routes, f"JS calls unknown routes: {urls - routes}")

    def test_static_files_and_page_links_load(self):
        for url in re.findall(r'(?:href|src)="(/static/[^"]+)"', self.html):
            self.assertEqual(self.client.get(url).status_code, 200, url)
        # url_for() output
        self.assertEqual(self.client.get("/static/css/style.css").status_code, 200)
        self.assertEqual(self.client.get("/static/js/app.js").status_code, 200)

    def test_html_has_no_external_scripts_or_old_mobilenet(self):
        self.assertNotRegex(self.html, r"<script[^>]+src=\"https?://")
        self.assertNotIn("mobilenet", (self.html + self.js).lower())
        self.assertNotIn("tensorflow", (self.html + self.js).lower())

    def test_no_manual_material_dropdown(self):
        self.assertNotIn("<select", self.html)

    def test_in_page_anchors_have_targets(self):
        ids = set(re.findall(r'id="([^"]+)"', self.html))
        for target in re.findall(r'href="#([^"]+)"', self.html):
            self.assertIn(target, ids, f"anchor #{target} has no target")


class TestDatabase(AppTestCase):
    def test_fresh_database_is_created_with_tables(self):
        self.assertTrue(os.path.exists(self.db_path))
        connection = sqlite3.connect(self.db_path)
        tables = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        connection.close()
        self.assertIn("scans", tables)
        self.assertEqual(self.client.get("/api/history").get_json(), [])
        stats = self.client.get("/api/stats").get_json()
        self.assertEqual((stats["total"], stats["recyclable"], stats["other_routes"], stats["needs_review"]), (0, 0, 0, 0))
        self.assertIsNone(stats["average_confidence"])

    def test_existing_old_prototype_database_does_not_crash_and_is_not_altered(self):
        path = os.path.join(self.tmp, "old", "echo_scan.db")
        os.makedirs(os.path.dirname(path))
        connection = sqlite3.connect(path)
        connection.execute("CREATE TABLE scan (id INTEGER PRIMARY KEY, item_name TEXT, category TEXT, "
                           "recyclable BOOLEAN, confidence INTEGER, guidance TEXT, created_at DATETIME)")
        connection.execute("INSERT INTO scan VALUES (1,'Banana','Food',0,80,'x','2026-01-01')")
        connection.commit()
        connection.close()
        app = create_app({"TESTING": True, "DATABASE": path}, classifier=FakeClassifier())
        client = app.test_client()
        self.assertEqual(client.get("/api/history").get_json(), [])           # old guesses are not shown
        self.assertEqual(client.get("/api/stats").get_json()["total"], 0)
        connection = sqlite3.connect(path)
        self.assertEqual(connection.execute("SELECT COUNT(*) FROM scan").fetchone()[0], 1)   # untouched
        connection.close()

    def test_reopening_the_same_database_keeps_data(self):
        self.assertEqual(self.scan().status_code, 201)
        app2 = create_app({"TESTING": True, "DATABASE": self.db_path}, classifier=FakeClassifier())
        self.assertEqual(len(app2.test_client().get("/api/history").get_json()), 1)


class TestImageValidation(AppTestCase):
    def assert_rejected(self, response, status):
        self.assertEqual(response.status_code, status, response.get_data(as_text=True))
        self.assertIn("error", response.get_json())
        self.assertEqual(self.fake.calls, 0, "model must not run on rejected input")
        self.assertEqual(self.client.get("/api/history").get_json(), [])

    def test_missing_file(self):
        response = self.client.post("/api/classify", data={}, content_type="multipart/form-data")
        self.assert_rejected(response, 400)

    def test_wrong_field_name(self):
        response = self.client.post("/api/classify", data={"file": (io.BytesIO(make_image()), "a.png")},
                                    content_type="multipart/form-data")
        self.assert_rejected(response, 400)

    def test_empty_file(self):
        self.assert_rejected(self.scan(b""), 400)

    def test_text_file_pretending_to_be_image(self):
        self.assert_rejected(self.scan(b"hello, I am not an image", "cat.jpg", "image/jpeg"), 415)

    def test_executable_pretending_to_be_image(self):
        self.assert_rejected(self.scan(b"MZ\x90\x00" + b"\x00" * 200, "virus.png", "image/png"), 415)

    def test_unsupported_but_real_image_format(self):
        self.assert_rejected(self.scan(make_image("GIF"), "a.gif", "image/gif"), 415)
        self.assert_rejected(self.scan(make_image("BMP"), "a.bmp", "image/bmp"), 415)

    def test_truncated_jpeg(self):
        self.assert_rejected(self.scan(make_image("JPEG", (400, 400))[:300], "t.jpg", "image/jpeg"), 415)

    def test_oversized_file(self):
        big = make_image("PNG") + b"\x00" * (9 * 1024 * 1024)
        self.assert_rejected(self.scan(big), 413)

    def test_huge_dimensions_rejected(self):
        buffer = io.BytesIO()
        Image.new("1", (9000, 9000)).save(buffer, "PNG")      # 81 million pixels, tiny file
        self.assertLess(len(buffer.getvalue()), 1024 * 1024)
        self.assert_rejected(self.scan(buffer.getvalue()), 413)

    def test_valid_formats_accepted(self):
        for fmt, name, mime in (("PNG", "a.png", "image/png"), ("JPEG", "a.jpg", "image/jpeg"),
                                ("WEBP", "a.webp", "image/webp")):
            self.assertEqual(self.scan(make_image(fmt), name, mime).status_code, 201, fmt)

    def test_rgba_and_palette_images_are_converted(self):
        buffer = io.BytesIO()
        Image.new("RGBA", (40, 40), (1, 2, 3, 128)).save(buffer, "PNG")
        self.assertEqual(self.scan(buffer.getvalue()).status_code, 201)

    def test_file_extension_is_ignored(self):
        # A real PNG named .txt is still accepted because the BYTES are checked, not the name.
        self.assertEqual(self.scan(make_image("PNG"), "notes.txt", "text/plain").status_code, 201)


class TestClassification(AppTestCase):
    def test_result_comes_from_classifier_and_is_saved(self):
        response = self.scan()
        self.assertEqual(response.status_code, 201)
        data = response.get_json()
        self.assertEqual(data["status"], "classified")
        self.assertEqual(data["predicted_label"], "Plastic")
        self.assertEqual(data["item"], "Plastic")
        self.assertAlmostEqual(data["confidence"], 0.93)                 # exactly the model's score
        self.assertEqual([a["label"] for a in data["alternatives"]], ["Glass", "Metal"])
        self.assertTrue(data["guidance"] and data["warning"])
        self.assertIn("Local recycling rules vary", data["warning"])
        self.assertIn("NOT", data["warning"])                            # plastic: not everything is recyclable
        self.assertEqual(self.fake.calls, 1)
        history = self.client.get("/api/history").get_json()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["id"], data["id"])
        self.assertEqual(history[0]["model_id"], "fake/test-model")

    def test_filename_and_extra_form_fields_cannot_change_the_result(self):
        for name in ("glass_bottle.png", "BATTERY.png", "banana.png"):
            data = self.scan(filename=name, extra={"category": "Battery", "material": "glass",
                                                   "label": "Metal"}).get_json()
            self.assertEqual(data["predicted_label"], "Plastic", name)

    def test_different_model_output_gives_different_guidance(self):
        self.fake.predictions = [{"label": "Battery", "probability": 0.97},
                                 {"label": "Metal", "probability": 0.02}]
        data = self.scan().get_json()
        self.assertEqual((data["item"], data["route"]), ("Battery", guidance.ROUTE_SPECIAL))
        self.fake.predictions = [{"label": "Biological", "probability": 0.9},
                                 {"label": "Paper", "probability": 0.05}]
        self.assertEqual(self.scan().get_json()["route"], guidance.ROUTE_COMPOST)

    def test_low_confidence_is_not_presented_as_confident(self):
        self.fake.predictions = [{"label": "Glass", "probability": 0.41},
                                 {"label": "Plastic", "probability": 0.33},
                                 {"label": "Metal", "probability": 0.20}]
        data = self.scan().get_json()
        self.assertEqual(data["status"], "low_confidence")
        self.assertEqual(data["category"], "Uncertain")
        self.assertEqual(data["route"], guidance.ROUTE_REVIEW)
        self.assertEqual(data["item"], "Not sure")
        self.assertEqual(data["recyclable"], "n/a")
        self.assertNotIn("is generally recyclable", data["guidance"])
        self.assertAlmostEqual(data["confidence"], 0.41)                 # real score still reported

    def test_close_call_between_two_classes_is_uncertain(self):
        self.fake.predictions = [{"label": "Paper", "probability": 0.66},
                                 {"label": "Cardboard", "probability": 0.60}]
        self.assertEqual(self.scan().get_json()["status"], "low_confidence")

    def test_unsupported_prediction_is_reported_honestly(self):
        self.fake.predictions = [{"label": "Spaceship", "probability": 0.99}]
        data = self.scan().get_json()
        self.assertEqual(data["status"], "unsupported")
        self.assertEqual(data["route"], guidance.ROUTE_REVIEW)

    def test_model_unavailable_returns_503_and_saves_nothing(self):
        self.fake.error = ClassifierUnavailable("The AI model is still loading.")
        response = self.scan()
        self.assertEqual(response.status_code, 503)
        self.assertIn("still loading", response.get_json()["error"])
        self.assertNotIn("predicted_label", response.get_json())
        self.assertEqual(self.client.get("/api/history").get_json(), [])

    def test_inference_error_returns_500_and_saves_nothing(self):
        self.fake.error = InferenceError("boom")
        response = self.scan()
        self.assertEqual(response.status_code, 500)
        self.assertIn("Nothing was saved", response.get_json()["error"])
        self.assertNotIn("confidence", response.get_json())
        self.assertEqual(self.client.get("/api/history").get_json(), [])

    def test_uploaded_image_is_not_stored_on_disk(self):
        self.scan()
        for folder, _dirs, files in os.walk(self.tmp):
            for name in files:
                self.assertNotRegex(name.lower(), r"\.(png|jpe?g|webp|gif|bmp)$", os.path.join(folder, name))
        with open(self.db_path, "rb") as handle:
            self.assertNotIn(b"\x89PNG", handle.read())


class TestHistoryStatsClear(AppTestCase):
    def add(self, label, probability, runner_up="Trash"):
        self.fake.predictions = [{"label": label, "probability": probability},
                                 {"label": runner_up, "probability": round((1 - probability) / 2, 3)}]
        self.assertEqual(self.scan().status_code, 201)

    def test_history_is_newest_first(self):
        self.add("Paper", 0.9)
        self.add("Glass", 0.9)
        self.add("Battery", 0.9)
        self.assertEqual([row["predicted_label"] for row in self.client.get("/api/history").get_json()],
                         ["Battery", "Glass", "Paper"])

    def test_stats_are_calculated_from_real_rows(self):
        self.add("Paper", 0.90)      # recycle
        self.add("Plastic", 0.80)    # recycle
        self.add("Battery", 0.95)    # special drop-off
        self.add("Clothes", 0.85)    # donate
        self.fake.predictions = [{"label": "Glass", "probability": 0.40}, {"label": "Metal", "probability": 0.30}]
        self.scan()                   # low confidence
        stats = self.client.get("/api/stats").get_json()
        self.assertEqual(stats["total"], 5)
        self.assertEqual(stats["recyclable"], 2)
        self.assertEqual(stats["other_routes"], 2)
        self.assertEqual(stats["needs_review"], 1)
        self.assertAlmostEqual(stats["average_confidence"], (0.90 + 0.80 + 0.95 + 0.85) / 4)
        self.assertEqual({row["category"]: row["count"] for row in stats["by_category"]},
                         {"Paper & cardboard": 1, "Plastic": 1, "Electronic waste / batteries": 1, "Textiles": 1})

    def test_clear_requires_confirmation(self):
        self.add("Paper", 0.9)
        for body in (None, {}, {"confirm": False}, {"confirm": "yes"}):
            response = self.client.delete("/api/history", json=body) if body is not None \
                else self.client.delete("/api/history")
            self.assertEqual(response.status_code, 400)
        self.assertEqual(len(self.client.get("/api/history").get_json()), 1)

    def test_clear_with_confirmation_deletes_everything(self):
        self.add("Paper", 0.9)
        self.add("Glass", 0.9)
        response = self.client.delete("/api/history", json={"confirm": True})
        self.assertEqual(response.get_json(), {"ok": True, "deleted": 2})
        self.assertEqual(self.client.get("/api/history").get_json(), [])
        self.assertEqual(self.client.get("/api/stats").get_json()["total"], 0)


class TestGuidanceLayer(unittest.TestCase):
    def test_every_model_class_has_guidance(self):
        self.assertEqual(set(guidance.GUIDANCE), set(classifier_module.EXPECTED_LABELS))

    def test_guidance_entries_are_complete_and_safe(self):
        routes = {guidance.ROUTE_RECYCLE, guidance.ROUTE_COMPOST, guidance.ROUTE_SPECIAL,
                  guidance.ROUTE_REUSE, guidance.ROUTE_GENERAL}
        for label, info in guidance.GUIDANCE.items():
            for field in ("item", "category", "route", "recyclable", "guidance", "warning"):
                self.assertTrue(info[field], f"{label}.{field} is empty")
            self.assertIn(info["route"], routes, label)
            self.assertIn(info["recyclable"], {"yes", "conditional", "no"}, label)
            self.assertIn("Local recycling rules vary", info["warning"], label)
        # No class claims that it is *always* recyclable.
        self.assertNotIn("yes", {info["recyclable"] for info in guidance.GUIDANCE.values()})
        self.assertEqual(guidance.GUIDANCE["Battery"]["route"], guidance.ROUTE_SPECIAL)
        self.assertEqual(guidance.GUIDANCE["Battery"]["recyclable"], "no")

    def test_interpret_rejects_empty_predictions(self):
        with self.assertRaises(ValueError):
            guidance.interpret([])


@unittest.skipIf(find_spec("torch") and find_spec("transformers"),
                 "torch is installed here; use tests/real_model_check.py instead")
class TestRealClassifierWithoutAiLibraries(unittest.TestCase):
    """Uses the REAL WasteClassifier class. Proves it fails honestly instead of faking a result."""

    def test_missing_libraries_raise_clear_error_and_no_prediction(self):
        real = WasteClassifier(cache_dir=tempfile.mkdtemp())
        with self.assertRaises(ClassifierUnavailable) as caught:
            real.predict(Image.new("RGB", (32, 32)))
        self.assertIn("pip install -r requirements.txt", str(caught.exception))
        self.assertEqual(real.status()["state"], classifier_module.STATE_ERROR)
        self.assertFalse(real.is_ready)

    def test_app_with_real_classifier_returns_503_not_a_fake_prediction(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp, True)
        app = create_app({"TESTING": True, "DATABASE": os.path.join(tmp, "db", "t.db")},
                         classifier=WasteClassifier(cache_dir=os.path.join(tmp, "models")))
        client = app.test_client()
        response = client.post("/api/classify", data={"image": (io.BytesIO(make_image()), "x.png")},
                               content_type="multipart/form-data")
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("predicted_label", response.get_json())
        self.assertEqual(client.get("/api/history").get_json(), [])


if __name__ == "__main__":
    unittest.main()
