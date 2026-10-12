# ECHO SCAN — AI Waste Classification & Recycling Guidance

ECHO SCAN is a Flask web app. You upload a photo (or take one with your camera), a **real pretrained waste-classification model** analyses the image on your own computer, and the app shows what the item is, where it generally goes, and how to dispose of it safely. Scans are saved to a local history with dashboard statistics.

There are **no accounts**: no login, registration or logout. Everything works straight away.

## How it works

```
Photo  ->  Flask server  ->  AI model (predicts a class)  ->  guidance.py (advice)  ->  SQLite history
```

| Part | File | Job |
|---|---|---|
| Web server and routes | `app.py` | Home page, `/api/classify`, history, stats, clear history |
| AI model | `classifier.py` | Loads the model and returns class probabilities |
| Recycling advice | `guidance.py` | Turns a class into category, route and instructions (edit this to change advice) |
| Image safety checks | `imaging.py` | Validates type and size, processes the image in memory |
| Database | `database.py` | Saves scan results in `instance/echo_scan.db` |
| Page | `templates/index.html`, `static/` | The user interface |

**The scan result always comes from the image.** There is no material dropdown; the file name and any other form data are ignored.

### The model

* **Model:** [`prithivMLmods/Augmented-Waste-Classifier-SigLIP2`](https://huggingface.co/prithivMLmods/Augmented-Waste-Classifier-SigLIP2) on Hugging Face (a SigLIP2 image-classification model fine-tuned for waste sorting).
* **Licence:** Apache-2.0.
* **Classes (10):** Battery, Biological, Cardboard, Clothes, Glass, Metal, Paper, Plastic, Shoes, Trash.
* **Version:** pinned to commit `d203b52311f70ac3245d5fb8345d88344bec138e`.
* **Size:** the weights are about 372 MB. They are **not** inside this ZIP; they are downloaded once and then cached in the `models` folder, after which the app works offline.
* The photo is processed in memory by your own Flask server and is **never saved**. It is not sent to any classification service. Only the result (label, category, confidence, advice, time) is stored.

## Requirements

* Windows 10/11, 64-bit.
* **64-bit Python 3.14.x** (this project is configured for your existing Python 3.14.5; it will not install or require another Python version).
* Internet for package installation and the **first model download** (about 372 MB for the model, plus Python packages). After the model is cached, inference is local and works offline.
* Chrome or Edge.

You already have Python 3.14.5. In PowerShell, verify it with `python --version`. If Windows reports another version, fix the existing Python PATH/launcher selection; this project does not install a second Python.

## Run it on Windows (PowerShell)

Extract the ZIP first. In PowerShell, go to the folder containing `app.py` and run:

```powershell
python --version
# Expected: Python 3.14.5
cd "$HOME\Downloads\ECHO_SCAN_Final\ECHO_SCAN"
.\run_windows.bat
```

If you extracted it somewhere else, replace the `cd` path with the folder that contains `app.py`. The batch file verifies Python 3.14, creates `.venv`, installs binary wheels only, downloads/tests the model, and starts Flask. It stops with an error if setup fails.

`run_windows.bat` creates `.venv`, installs the packages, downloads the model, then starts the app and opens <http://127.0.0.1:5000>. Keep the window open while you use the app; press `Ctrl+C` to stop it. The first run takes several minutes; later runs start in seconds.

### Doing it step by step instead

```powershell
cd "$HOME\Downloads\ECHO_SCAN_Final\ECHO_SCAN"
python --version
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install --only-binary=:all: -r requirements.txt
.\.venv\Scripts\python.exe setup_model.py
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
.\.venv\Scripts\python.exe app.py
```

You do not need to activate the virtual environment, so PowerShell execution-policy changes are unnecessary.

## Using the app

1. **Upload image** (JPG, PNG or WebP, up to 8 MB) **or Start camera → Capture photo**.
2. Check the preview, then press **Scan with AI**.
3. Read the result: the predicted item, where it goes, the model's confidence, what to do, and a *check local rules* note.
4. Results are saved to **Recent scans**; the cards at the top show real totals. **Clear history** asks for confirmation first.

The camera only works at `http://127.0.0.1:5000` or `http://localhost:5000`, and your browser will ask permission. If there is no camera, use Upload.

### What the result means

* **Confidence** is the model's own probability score for its top class. It is not a guarantee.
* If the model is unsure (top score below 60%, or too close to the second guess), ECHO SCAN shows **"Needs review"** and gives no sorting recommendation. Retake the photo with one item in good light on a plain background.
* "Recyclable" always means *generally recyclable if clean and accepted locally*. Contaminated paper, plastic bags, multilayer wrappers, mirrors and similar items are called out in the warnings because they are often rejected.
* If the model cannot run (not installed, still downloading, error), the app says so and **shows no result**. It never makes up a prediction.

## Testing

Automated tests (no model download needed; they use a stand-in classifier for the website, database and validation logic):

```powershell
python -m unittest discover -s tests -v
```

To check the **real model** on your own photos (needs the model downloaded):

```powershell
python tests\real_model_check.py photo1.jpg photo2.png
python tests\real_model_check.py --folder sample_images
```

For `--folder`, put photos in sub-folders named after the classes (for example `sample_images\Glass\jar1.jpg`). It prints each prediction and your accuracy.

## Known limitations

* The model has **10 fixed classes** and always picks one, even for a photo of something unrelated (a face, a desk). The confidence threshold catches many such cases but not all. Scan one item at a time.
* The model card reports ~99% accuracy on its own test split, but does not name its training dataset. Photos from your own phone, with clutter or odd lighting, can score lower. Use `real_model_check.py` to measure it on your photos.
* "Plastic" and "Paper" do not tell the type (for example PET vs a snack wrapper). The advice is deliberately cautious. "Trash" is a catch-all for items that fit none of the other classes.
* Disposal advice is general. Local rules vary and always win.
* Scan history from the older prototype (`scan` table, created by a general MobileNet model) is left untouched in `instance/echo_scan.db` but is not shown, so old guesses are not mixed with real model results.
* The page uses Google Fonts when online; offline it falls back to a standard font.

## Troubleshooting

| Problem | Fix |
|---|---|
| Python 3.14 check fails | Run `python --version`. Make sure the existing Python 3.14 installation is on PATH. This project will not install another Python version. |
| Package install fails | Check internet and that Python is 64-bit. The installer uses `--only-binary=:all:` to prevent source builds. Read the first pip error; if needed, delete `.venv` and retry. |
| "The AI model could not be loaded" | Check internet for the first download, then restart. Delete the `models` folder if a download was interrupted. |
| Page says the model is still loading | Wait; the first start loads the model into memory. Progress shows at the bottom of the result panel. |
| Port 5000 already in use | Close the other program using port 5000, or change `port=5000` at the bottom of `app.py`. |
