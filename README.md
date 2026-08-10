# Maintenance AI Matching System — Image ↔ Arabic Text Verification

A two-input verification system that checks whether an **uploaded maintenance image** actually matches an **Arabic problem description**, returning `MATCH` / `MISMATCH` / `UNCERTAIN`. Built as the core AI feature of a graduation project (a smart app connecting customers with maintenance technicians) — implemented **twice**, with two independent approaches: a **classical ML pipeline** and a **deep learning pipeline**.

## Why this exists

When a customer submits a maintenance request, they upload a photo and describe the problem in Arabic. Before the request reaches a technician, the system verifies the photo genuinely shows the described problem (e.g. rejecting an irrelevant photo, or catching a plumbing photo paired with an electrical complaint) — reducing fraud/spam and misrouted jobs.

## Two Implementations

| | Classical ML Pipeline | Deep Learning Pipeline |
|---|---|---|
| Notebooks | 4 notebooks (`notebooks/ml_pipeline/`) | 1 notebook (`notebooks/dl_pipeline/`) |
| Visual stage | Handcrafted features (HOG/LBP/color/texture) + XGBoost/SVM ensemble + one-vs-rest verifiers | Deep CNN classifiers (ResNet50 best, vs. EfficientNet-B0, MobileNetV3, YOLO11-cls) |
| Text stage | TF-IDF + engineered features + WOA-optimized Stacking ensemble | MiniLM embeddings + TF-IDF + keyword features, hybrid ML fusion |
| Decision layer | Rule-based fusion across 3 stages (1A relevance → 1C category verification → Stage 2 text) | Rule-based fusion across 2 stages (image category vs. text category) |

Both pipelines solve the same problem end-to-end; they exist as parallel experiments to compare a lightweight classical-ML approach against a deep-learning approach for the same task.

---

## Classical ML Pipeline

![Full ML System Pipeline](assets/ml_full_pipeline_flowchart.png)

| Notebook | Stage | What it does |
|---|---|---|
| `01_stage1a_image_relevance_detection.ipynb` | Stage 1A | Filters out fake/irrelevant images before deeper analysis. Handcrafted visual features (26,419-dim) + **XGBoost + SVM ensemble** (0.75/0.25 weights), evaluated with stratified k-fold CV. Outputs `RELEVANT` / `IRRELEVANT` / `UNCERTAIN`. |
| `02_stage1c_visual_category_verification.ipynb` | Stage 1C | Given an image already predicted `RELEVANT` and a category from Stage 2 text, visually verifies whether the image supports that category. Uses one-vs-rest verifiers per category + a pairwise specialist classifier + a weighted fusion score (general verifier 0.40, multiclass 0.25, pairwise 0.35). |
| `03_stage2_arabic_text_matching.ipynb` | Stage 2 | Classifies the Arabic problem description into a maintenance category/subcategory. Compares Random Forest, Extra Trees, Gradient Boosting, Logistic Regression, Soft Voting, and a **WOA (Whale Optimization Algorithm) — optimized Stacking ensemble**, which wins overall. |
| `04_production_inference_and_decision_engine.ipynb` | Fusion | Packages Stages 1A/1C/2 into a clean production inference package and combines their outputs into the final `MATCH` / `MISMATCH` decision, with early rejection for irrelevant images. |

### ML Pipeline Results

**Stage 1A — Image Relevance Detection** (5,026 images; best: XGBoost + SVM ensemble)
![Stage 1A Results](assets/ml_stage1a_results.png)
- Cross-validation: 95.47% accuracy, 0.9881 ROC AUC
- Final test: 93.64% accuracy, 92.10% F1
- Tri-state output coverage: 88.57% (accept precision 91.20%, reject precision 98.96%)

**Stage 1C — Visual Category Verification** (2,005 relevant images; one-vs-rest per category)
![Stage 1C Results](assets/ml_stage1c_results.png)
- Strongest category verifier: Painting (نقاشة) — 87.03% accuracy, 0.9419 ROC AUC
- Most challenging category: Plumbing (سباكة) — high precision (83.33%) but low recall (7.04%) as a standalone verifier
- Fusion combines a general verifier, a multiclass score, and pairwise specialist scores

**Stage 2 — Arabic Text Matching** (3,888 texts, 21,143 pairs after augmentation; best: WOA Optimized Stacking V3)
![Stage 2 Results](assets/ml_stage2_results.png)
- Best holdout results: 98.37% accuracy, 97.82% F1, 0.9977 ROC AUC

---

## Deep Learning Pipeline

![Full DL System Pipeline](assets/dl_full_pipeline_flowchart.png)

A single end-to-end notebook (`dl_full_system_stage1_stage2_decision_engine.ipynb`) implementing all 3 stages:

1. **Stage 1 — Visual DL Classifier**: image → {plumbing, electricity, carpentry, painting, irrelevant}. Compares YOLO11n-cls, YOLO11s-cls, EfficientNet-B0, MobileNetV3-Large-100, and **ResNet50** (winner) via `timm`, trained with PyTorch.
2. **Stage 2 — Arabic Text Classifier v2**: Arabic description → category, using MiniLM embeddings + word/char TF-IDF + keyword features, comparing Logistic Regression, Calibrated Linear SVC (winner), and SGD (log-loss).
3. **Stage 3 — Decision Engine**: rule-based fusion — irrelevant image → early reject; non-maintenance text → mismatch; matching categories → `MATCH`; differing categories → `MISMATCH`/`UNCERTAIN` by confidence.

Also includes a Gradio interface for interactive testing with image + Arabic text input.

### DL Pipeline Results

**Stage 1 — Visual DL Classifier** (4,958 images; best: ResNet50)
![DL Stage 1 Results](assets/dl_stage1_results.png)
- 94.76% accuracy, 90.18% macro F1, **98.90% irrelevant-image recall** (1.10% false accept rate)

**Stage 2 — Arabic Text Classifier v2** (5,903 texts; best: Calibrated Linear SVC)
![DL Stage 2 Results](assets/dl_stage2_results.png)
- 94.59% test accuracy, 94.75% macro F1, **100% on a 30-case user-style stress test**

**Overall System Summary**
![DL Overall Summary](assets/dl_overall_summary.png)

---

## Tech Stack

**Classical ML:** scikit-image (HOG/LBP/GLCM features), XGBoost, scikit-learn (SVM, ensembles, TF-IDF), a Whale Optimization Algorithm (WOA) for stacking-ensemble hyperparameter tuning
**Deep Learning:** PyTorch, `timm` (ResNet50, EfficientNet, MobileNetV3), Ultralytics YOLO11-cls
**NLP:** sentence-transformers (multilingual MiniLM), TF-IDF
**UI:** Gradio

## Project Structure

```
├── notebooks/
│   ├── ml_pipeline/
│   │   ├── 01_stage1a_image_relevance_detection.ipynb
│   │   ├── 02_stage1c_visual_category_verification.ipynb
│   │   ├── 03_stage2_arabic_text_matching.ipynb
│   │   └── 04_production_inference_and_decision_engine.ipynb
│   └── dl_pipeline/
│       └── dl_full_system_stage1_stage2_decision_engine.ipynb
├── assets/            (results tables & flowcharts)
├── requirements.txt
└── README.md
```

## Getting Started

```bash
pip install -r requirements.txt
```

Each notebook expects the relevant dataset/artifact paths (image folders, `.csv` label files, prior-stage `.joblib`/model checkpoints) as configured in its early cells — see in-notebook comments for exact expected paths (originally run on Google Colab with Drive-mounted data).

## Author

Nourhan — AI undergraduate, New Mansoura University. Core AI verification feature of a graduation project building a smart app connecting customers with maintenance technicians.
