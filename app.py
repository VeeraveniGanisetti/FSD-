"""ECHO SCAN - classify waste photos with a real AI model and get recycling guidance.

There are no accounts: every feature works straight away.

Run it with:   python app.py      then open http://127.0.0.1:5000
"""
import os

from flask import Flask, g, jsonify, render_template, request

import database
import guidance
from classifier import ClassifierUnavailable, InferenceError, WasteClassifier
from imaging import MAX_UPLOAD_BYTES, ImageRejected, load_image

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


def create_app(test_config=None, classifier=None):
    """Build the Flask app. `classifier` can be replaced in automated tests."""
    app = Flask(__name__)
    app.config.update(
        DATABASE=os.path.join(BASE_DIR, "instance", "echo_scan.db"),
        MAX_CONTENT_LENGTH=MAX_UPLOAD_BYTES + 512 * 1024,   # room for the form wrapper
    )
    if test_config:
        app.config.update(test_config)

    os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)
    database.init_db(app.config["DATABASE"])
    app.classifier = classifier or WasteClassifier()

    # ------------------------------------------------------------ helpers
    def get_db():
        if "db" not in g:
            g.db = database.connect(app.config["DATABASE"])
        return g.db

    @app.teardown_appcontext
    def close_db(_error):
        connection = g.pop("db", None)
        if connection is not None:
            connection.close()

    @app.after_request
    def security_headers(response):
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Permissions-Policy"] = "camera=(self), microphone=()"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; "
            "style-src 'self' https://fonts.googleapis.com; font-src https://fonts.gstatic.com; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; base-uri 'self'; frame-ancestors 'none'"
        )
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    # ------------------------------------------------------------- pages
    @app.get("/")
    def home():
        return render_template("index.html")

    # --------------------------------------------------------------- API
    @app.get("/api/health")
    def health():
        return jsonify({"ok": True, "model": app.classifier.status()})

    @app.post("/api/classify")
    def classify():
        """Take an image, run the AI model on it, save the result, return the guidance."""
        upload = request.files.get("image")
        if upload is None:
            return jsonify({"error": "No image was received. Choose or capture a photo first."}), 400

        try:
            image = load_image(upload.read(MAX_UPLOAD_BYTES + 1))
        except ImageRejected as error:
            return jsonify({"error": error.message}), error.status

        try:
            predictions = app.classifier.predict(image, top_k=3)
        except ClassifierUnavailable as error:
            return jsonify({"error": str(error), "model": app.classifier.status()}), 503
        except InferenceError:
            app.logger.exception("Inference failed")
            return jsonify({"error": "The AI model could not analyse this image. Nothing was saved. "
                                     "Try another photo."}), 500

        result = guidance.interpret(predictions)
        scan_id, created_at = database.save_scan(get_db(), result, app.classifier.model_id)
        result.update({"id": scan_id, "created_at": created_at, "model_id": app.classifier.model_id})
        return jsonify(result), 201

    @app.get("/api/history")
    def history():
        return jsonify(database.list_scans(get_db()))

    @app.get("/api/stats")
    def stats():
        return jsonify(database.stats(get_db()))

    @app.delete("/api/history")
    def clear_history():
        body = request.get_json(silent=True) or {}
        if body.get("confirm") is not True:
            return jsonify({"error": "Clearing history needs confirmation."}), 400
        deleted = database.clear_scans(get_db())
        return jsonify({"ok": True, "deleted": deleted})

    # ------------------------------------------------------ error handlers
    def wants_json():
        return request.path.startswith("/api/")

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify({"error": "That image is too large. Choose a photo smaller than 8 MB."}), 413

    @app.errorhandler(404)
    def not_found(_error):
        if wants_json():
            return jsonify({"error": "Not found."}), 404
        return "Page not found. <a href='/'>Back to ECHO SCAN</a>", 404

    @app.errorhandler(405)
    def wrong_method(_error):
        return jsonify({"error": "That action is not allowed here."}), 405

    @app.errorhandler(500)
    def server_error(_error):
        return jsonify({"error": "Something went wrong on the server."}), 500

    return app


if __name__ == "__main__":
    flask_app = create_app()
    flask_app.classifier.warm_up_async()       # start loading the model in the background
    flask_app.run(host="127.0.0.1", port=5000, debug=False)
