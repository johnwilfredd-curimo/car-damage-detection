# Car Damage Detection

[![API tests](https://github.com/johnwilfredd-curimo/car-damage-detection/actions/workflows/api-tests.yml/badge.svg)](https://github.com/johnwilfredd-curimo/car-damage-detection/actions/workflows/api-tests.yml)

Classifies a photo of the front or rear of a car as **normal, broken or crushed**: six classes, from a
fine-tuned ResNet50, served as a Streamlit app and a FastAPI service.

**[Live demo](https://car-damage-detection-jwc.streamlit.app/)** ·
**[API docs](https://car-damage-api.srv1818955.hstgr.cloud/docs)** ·
**[Notebook](https://nbviewer.org/github/johnwilfredd-curimo/car-damage-detection/blob/main/notebooks/car_damage_detection.ipynb)**

---

## The problem

A proof of concept for a used-car business: given a photo of a car taken from a front or rear corner, say which
end it is and what condition it is in. The brief set one acceptance number, **accuracy above 75%**, on a
dataset of only **2,300 labelled images**.

That dataset size is the whole problem. It is far too little to teach a network to see from scratch, and the
first two models prove it.

## Result

| # | Model | What trains | Validation accuracy |
|---|---|---|---|
| 1 | CNN built by hand | everything, from random weights | 56.81% |
| 2 | Same CNN + BatchNorm, Dropout, weight decay | everything | 57.68% |
| 3 | EfficientNet-B0, pretrained, frozen | the classifier only | 66.09% |
| 4 | ResNet50, pretrained, last block unfrozen | `layer4` + the classifier | 83.19% |
| 4b | ResNet50 after Optuna tuning | same | **84.64%** |

Each figure is the best of 10 epochs on the 345-image validation set. Optuna searched the learning rate and
the dropout rate over 20 trials of 3 epochs each (10 pruned early) and settled on `lr` 0.0012 and dropout
0.39. Everything was trained on a CPU.

**Held-out test set** (345 images the model never saw during training or tuning):

| Metric | Value |
|---|---|
| Accuracy | **80.29%** against a brief of 75% |
| Damaged cars correctly flagged as damaged | **193 of 225** (recall 85.8%) |

The test figure is about four points under validation. That is expected: the validation set chose the
hyperparameters and the epoch, so it flatters the model. The test set chose nothing, which makes 80.29% the
number to trust.

| Class | Precision | Recall | F1 | Test images |
|---|---|---|---|---|
| Front Breakage | 0.87 | 0.88 | 0.87 | 75 |
| Front Crushed | 0.85 | 0.57 | 0.68 | 60 |
| Front Normal | 0.76 | 0.97 | 0.85 | 75 |
| Rear Breakage | 0.82 | 0.80 | 0.81 | 45 |
| Rear Crushed | 0.71 | 0.64 | 0.67 | 45 |
| Rear Normal | 0.81 | 0.87 | 0.84 | 45 |

### Why regularisation did not help, and transfer learning did

Model 1, learning from random weights, climbs slowly and finishes 10 epochs at 56.81%, nowhere near the
brief. The textbook next step is regularisation, so Model 2 adds BatchNorm, Dropout and weight decay. It
**barely moves**: 57.68% against 56.81%, and its validation accuracy swings between 47.83% and 57.68% over the
last six epochs. Regularisation controls overfitting; it cannot supply information that 1,610 training images
do not contain.

The fix was to stop learning to see from scratch. Starting from ImageNet weights and training only the
classifier (Model 3) beat both hand-built networks. Unfreezing the last residual block as well (Model 4), so
the high-level features could adapt to damage, is what cleared the brief.

### What the accuracy number hides

Accuracy treats every mistake alike. For a business that inspects cars, calling a crushed bumper "normal" is a
different kind of error from mixing up "breakage" and "crushed". So the notebook also reports the six classes
collapsed to damaged / not damaged, and the recall on damaged.

On the test set, the model caught 193 of 225 damaged cars. What the confusion matrix shows about the rest:

- **Front and rear are almost never confused**: 2 images out of 345.
- **The costly error is concentrated in one class.** Of the 32 damaged cars called normal, 18 are crushed
  fronts. Recall on Front Crushed is 0.57, the weakest figure in the table. Rear damage called normal
  accounts for 9, and front breakage for 5.
- **Breakage versus crushed** is the other large group: 26 images were given the right end of the car and
  the wrong degree of damage.

So the accuracy clears the brief, but a business relying on this model would need to know that it waves
through roughly three in ten crushed fronts.

### What I changed from the course baseline

| Course baseline | This rebuild | Why it matters |
|---|---|---|
| One transform pipeline for all data | Augmentation on training data only | Validation was being measured on randomly altered images |
| Unseeded 75 / 25 random split | Seeded, stratified 70 / 15 / 15 | Runs are comparable; every class is represented in every slice |
| No test set | A test set touched once, at the end | The reported number was not used to choose the model |
| Keeps the last epoch's weights | Keeps the best epoch's weights | In the final run epoch 7 scored 84.64% and epoch 10 scored 82.90% |
| Returns a label | Returns a label, a confidence and all six probabilities | A 60% guess no longer looks like a 99% call |

## Architecture

One model, two interfaces:

```
src/prediction.py   ← builds the network, loads the weights, classifies image bytes
   ├── streamlit_app.py   human-facing demo
   └── api/main.py        machine-facing JSON service
```

Both import the same `predict()`. There is one copy of the preprocessing and one copy of the class list, so the
demo and the API cannot drift apart.

The weights are 94 MB, so they are **not in git**. They are published as a
[release asset](https://github.com/johnwilfredd-curimo/car-damage-detection/releases) and downloaded once, on
first start (Streamlit) or at image build (Docker). Inference runs on CPU.

```
car-damage-detection/
├─ streamlit_app.py          Streamlit entrypoint
├─ src/prediction.py         shared prediction module
├─ api/main.py               FastAPI service
├─ postman/                  Postman collection, fixtures, run screenshot
├─ samples/                  demo images (see samples/CREDITS.md)
├─ artifacts/                weights are downloaded here; not committed
├─ dataset/                  course dataset, not committed (see Data)
├─ notebooks/                the full end-to-end notebook
└─ Dockerfile                serves the API on port 7860
```

## Run it locally

```bash
python -m venv .venv
source .venv/bin/activate                 # macOS / Linux
# Windows Git Bash:    source .venv/Scripts/activate
# Windows PowerShell:  .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

The first start downloads the weights into `artifacts/`.

The API, from the repo root:

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

Then open <http://127.0.0.1:8000/docs>.

```bash
curl -F "file=@postman/fixtures/car.jpg" http://127.0.0.1:8000/predict
```

```json
{
  "prediction": "Front Normal",
  "confidence": 0.8292,
  "probabilities": {
    "Front Breakage": 0.1017, "Front Crushed": 0.0433, "Front Normal": 0.8292,
    "Rear Breakage": 0.0051, "Rear Crushed": 0.0042, "Rear Normal": 0.0164
  }
}
```

Bad input is rejected with a status code that says why: **415** for a file that is not a JPEG or PNG, **413**
over 5 MB, **400** for a file that claims to be an image and is not, **422** for a request with no file.

## API tests

`postman/` holds a Postman collection covering the health check, a valid upload, and three rejected uploads.
Beyond the response shape, it asserts an **invariant**: the returned `prediction` and `confidence` must always
be the highest entry in `probabilities`, whatever the model decides. `base_url` is a collection variable
defaulting to `http://127.0.0.1:8000`, so from the repo root it runs with no setup:

```bash
npx newman run postman/car-damage-detection.postman_collection.json
```

Against a deployed instance, override that one variable:

```bash
npx newman run postman/car-damage-detection.postman_collection.json \
    --env-var base_url=https://<host>
```

## Limits

- Trained on the course's 2,300 images, all taken from a front or rear corner. A side view, an
  interior, or a photo that is not a car still gets one of the six labels. Check the confidence.
- **It is weaker on photos from anywhere else.** The five images in `samples/` come from Wikimedia Commons,
  not from the training source. The model labels the two clearest ones correctly, calls both crushed cars
  "breakage", and reads the undamaged rear as a front at 41% confidence. The 80% test accuracy describes
  photos like the training set, not photos in general.
- Crushed fronts are missed often (see above), and "breakage" versus "crushed" is a judgement call in the
  labels.
- A proof of concept, as the brief specified. Not validated for any real inspection or claims decision.

## Data

The dataset is not in this repository. It is part of the exercise files of the Codebasics course, and
Codebasics does not permit redistributing them.

To re-run the notebook, enrol in the course, download the Section 14 exercise files, and place the six class
folders in `dataset/` at the repo root:

    dataset/F_Breakage/   dataset/F_Crushed/   dataset/F_Normal/
    dataset/R_Breakage/   dataset/R_Crushed/   dataset/R_Normal/

The live demo and the API need none of this: they use only the trained weights. Images under `samples/` and
`postman/fixtures/` are not from the course; their sources are listed in `samples/CREDITS.md`.

## Attribution

Project brief, dataset and baseline approach from the Codebasics course *Deep Learning: Beginner to Advanced*
(codebasics.io). This repository is my own end-to-end rebuild: the lifecycle structure, the data split and
evaluation design, the analysis write-up, the serving layer, API, test suite and deployment are mine. The
published weights are from my own training run.

Self-directed learning project; not commissioned client work. The scenario is the course's: "VROOM Cars" is
the client named in its brief, and AtliQ Technologies is a real technology consultancy, co-founded by the
course instructor, which the course casts as the service provider. I have no affiliation with either, or with
any other business of a similar name, and none of this work was done for them.
