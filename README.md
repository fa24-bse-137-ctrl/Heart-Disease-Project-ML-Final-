# CardioCheck: Heart Disease Prediction System

A machine learning web app that predicts whether a patient is likely to have heart disease from **4 inputs**. Built with scikit-learn (SVC) and Flask, ready to deploy to the cloud.

---

## What is in this folder

| File / folder | What it is |
|---|---|
| `train_model.py` | The ML code (the Titanic code, modified). Trains, tests and saves the model. |
| `streamlit_app.py` | The web app for **Streamlit Community Cloud** (same model, same design). |
| `app.py` | The same web app built with Flask, for hosts like Render. |
| `.streamlit/config.toml` | Streamlit colour theme. |
| `data/` | The 4 hospital data files you were given. |
| `model/svc_trained_model.pkl` | The trained model (created by `train_model.py`). |
| `outputs/` | Results: accuracy, confusion matrix picture, predictions, training log. |
| `requirements.txt` | The Python libraries the project needs. |
| `Procfile`, `render.yaml`, `.python-version` | Settings the cloud host reads when deploying. |

---

## Part 1: Run it on your computer

You need Python 3.10 or newer. Open a terminal inside this folder.

```bash
# 1. (Recommended) create a clean environment
python -m venv venv
venv\Scripts\activate            # Windows
# source venv/bin/activate       # Mac / Linux

# 2. Install the libraries
pip install -r requirements.txt

# 3. Train the model (prints every step, saves the model)
python train_model.py

# 4. Start the website
python app.py
```

Open **http://localhost:5000** in your browser.

To run the Streamlit version instead: `streamlit run streamlit_app.py`.

---

## Part 2: What the code does

### The 4 attributes I chose

| Attribute | Meaning | Values |
|---|---|---|
| `sex` | Sex of the patient | 1 = male, 0 = female |
| `cp` | Chest pain type | 1 typical angina, 2 atypical angina, 3 non-anginal pain, 4 no symptoms |
| `exang` | Chest pain during exercise | 1 = yes, 0 = no |
| `oldpeak` | ST depression on the exercise ECG | a number, about -3 to 7 |

I tried many combinations of 4 and picked this one because it scored best among options that keep most of the patients (857 of 920) and uses questions that are easy to answer.

### The steps (same order as the Titanic code)

1. **Import libraries** - tools for data (pandas), maths (numpy), ML (scikit-learn) and charts (matplotlib).
2. **Load data** - reads the 4 hospital files and joins them into one table of 920 patients.
3. **Understand data / pick 4 attributes** - keeps only the 4 inputs and the answer column. The original answer (`num`) is 0 for healthy and 1 to 4 for disease; I turned it into 0 = healthy, 1 = disease. Patients with a missing value in the 4 columns are removed (63 rows).
4. *(Titanic step 4 is not needed here.)*
5. **Encoding - skipped.** In Titanic this turned words like "male" into numbers. The heart data is already all numbers, so no label encoding is needed, as your teacher said.
6. **Train** - splits the data 80% / 20% (685 patients to learn from, 172 hidden for testing), then trains an **SVC (Support Vector Classifier)** and saves it to `model/svc_trained_model.pkl`. I added a `StandardScaler` before it. `oldpeak` can reach 7 while the other inputs are 0 or 1, and SVMs work better when inputs are on a similar scale.
7. **Test** - loads the saved model, predicts the 172 hidden patients, and prints accuracy, a classification report and a confusion matrix (also saved as `outputs/confusion_matrix.png`).
8. **Predict for a new patient** - edit the 4 values near the bottom of `train_model.py` to try a new patient.
9. **Improvements** - ideas to make the model better.

### Results

The model gets about **83% accuracy** on the 172 test patients. In the confusion matrix, 88 disease cases were correctly found, 6 were missed, 55 healthy patients were correctly cleared and 23 were wrongly flagged.

### How the website works

1. You fill in the form and press **Check result**.
2. The page (served by `app.py`) sends your 4 answers to `/predict`.
3. `app.py` checks the answers are valid, loads the saved model and calls `model.predict(...)`.
4. The result goes back to the page, which draws the ECG trace and shows the outcome.

---

## Part 3a: Deploy on Streamlit Community Cloud (free)

1. Upload the project to a public GitHub repository (see Part 3b, Step 1).
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click **Create app**, then **Deploy a public app from GitHub**.
4. Repository: choose your repo. Branch: `main`. **Main file path: `streamlit_app.py`**.
5. Click **Deploy** and wait a few minutes. You get a link ending in `.streamlit.app`.

## Part 3b: Deploy on Render (free, Flask version)

"Deploying" means putting your project on a computer that is always on, so anyone can open it with a link. Render is free and needs no credit card.

### Step 1 - Put the project on GitHub
1. Create a free account at https://github.com.
2. Click **New repository**. Name it `heart-disease-project` (no spaces). Keep it **Public**. Click **Create repository**.
3. On the new repository page click **uploading an existing file**.
4. Open this folder on your computer, select **everything inside it** (not the folder itself) and drag it into the page.
5. Wait for the upload, then click **Commit changes**.

Check that you can see `app.py`, `requirements.txt` and `train_model.py` directly on the repository's main page.

### Step 2 - Create the Render service
1. Create a free account at https://render.com (you can sign up with GitHub).
2. Click **New +** then **Web Service**.
3. Connect your GitHub and select the `heart-disease-project` repository.
4. Fill in the settings:
   - **Language:** Python 3
   - **Build Command:** `pip install -r requirements.txt && python train_model.py`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Under **Environment Variables** add `PYTHON_VERSION` = `3.12.3`.
6. Click **Create Web Service**.

### Step 3 - Wait and open your link
Render installs the libraries, trains the model and starts the server. This takes 3 to 6 minutes. Watch the log until you see **Your service is live**. Your link looks like `https://heart-disease-project.onrender.com`. Open it and submit a prediction. Share this link with your teacher.

### Notes
- **Free plan sleeps.** After about 15 minutes with no visitors the site sleeps, and the next visit takes 30 to 60 seconds to wake up. Open the link yourself before showing it to anyone.
- **To update the site**, change a file on GitHub. Render redeploys automatically.
- **Alternative:** the same files work on PythonAnywhere, Railway or Hugging Face Spaces.

### If something goes wrong
| Problem | Fix |
|---|---|
| Build fails with "No such file requirements.txt" | You uploaded the folder instead of its contents. Re-upload the files so `requirements.txt` is in the repository root. |
| "Application failed to respond" | Check the Start Command is exactly `gunicorn app:app`. |
| Page loads but Check result does nothing | Open the browser console (F12) and read the red error, or check the Render **Logs** tab. |
| `ModuleNotFoundError` on your computer | Run `pip install -r requirements.txt` again inside your activated environment. |

---

Dataset: UCI Heart Disease Databases. Principal investigators: Andras Janosi, M.D. (Hungarian Institute of Cardiology, Budapest); William Steinbrunn, M.D. (University Hospital, Zurich); Matthias Pfisterer, M.D. (University Hospital, Basel); Robert Detrano, M.D., Ph.D. (V.A. Medical Center, Long Beach and Cleveland Clinic Foundation).

*Predictions are informational and are not a medical diagnosis.*
