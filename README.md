# Recall Engine 

![CI](https://github.com/arjandhinsa/recall-engine/actions/workflows/ci.yml/badge.svg)

**Predicting which patients will fail to return for their recommended eye examination, using real practice data, then translating that prediction into recovered revenue.**

Currently being extended from a churn model into a deployed recall prioritisation engine: expected-revenue-ranked outreach (churn × dispense value), served as a monitored API.

An end-to-end machine learning project built on ~38,000 patients' worth of real appointment, dispensing, and prescription data from an independent optometry practice. The headline result is a temporally-validated churn model (AUC 0.63) that identifies **1.38× more at-risk patients than random outreach**, worth an estimated **£660 in recovered revenue per recall cycle** on held-out data and then quantified against the practice's own margins.

> The most substantial work in this project was not the model. It was diagnosing and rejecting **two structurally-flawed churn definitions** before arriving at a label that is business-truthful, leakage-free, and consistent with how the model would score in live deployment. That reasoning chain is documented below.

---

## Live demo

**https://recall.seyn.co.uk/docs** — the API running on AWS, on synthetic data.

- `GET /recall-list?top_n=20` — the ranked recall list
- `POST /score` — churn probability, expected value and priority for one patient
- `GET /health`

Runs locally too: `docker compose up`


## Run the demo

The full service runs on synthetic data, so no real patient data is required
or included:

```bash
docker compose up
```

Then open http://localhost:8000/docs for the interactive API:

- `POST /score` — churn probability, expected value and priority for one patient
- `GET /recall-list?top_n=20` — the ranked recall list
- `GET /health`

To regenerate the synthetic practice and retrain from scratch:

```bash
python -m src.make_synthetic
DATA_DIR=data_working/synthetic python -m src.pipeline
DATA_DIR=data_working/synthetic python -m src.batch_score
```



## Problem

Independent optometry practices lose measurable revenue when patients drift away between recall cycles. A lapsed patient means a missed sight test *and* the associated dispense (glasses or contact lenses). Recalls are the practice's core retention mechanism, but outreach is limited: staff can only call so many patients. The question this project answers:

**Given a patient's history, can we predict who is about to churn, accurately enough to focus limited outreach on the patients who would actually have lapsed?**

If the model concentrates real churners into a small, contactable list, a fixed outreach budget recovers more revenue than contacting patients at random.

---

## Data

Four raw exports from the practice's VisionPlus management system:

| Table | Rows | Content |
|---|---|---|
| Patients | 38,379 | Demographics (pseudonymised) |
| Appointments | 39,040 | Every appointment, with attendance status |
| Orders | 127,222 | Dispensing lines — frames, lenses, coatings, with value |
| Prescriptions | 30,005 | Clinical data, including the recall interval (`ReaMonths`) and projected recall date |

**Data governance.** All patient-identifying fields (names, addresses, full postcodes, dates of birth) are removed or reduced before any analysis. Patient references are converted to non-reversible salted HMAC-SHA256 keys, with the salt held outside the repository. All raw and working data is excluded from version control; nothing patient-identifying is committed at any point.

---

## Approach

### Building an honest churn label

Defining "churn" on a fixed historical extract is deceptively hard, and getting it wrong produces a model that scores well and means nothing. The label went through three iterations:

**Attempt 1 - "overdue past the projected recall date."** This produced an implausible **84.5% churn rate**. Investigation revealed the recall date field was a *projected* next-due date (last visit + interval), not a recall the patient had responded to, so a fixed window after it systematically missed patients who returned early or late.

**Attempt 2 - "gap since last visit exceeds the recall interval."** This gave **65%**, then swung to **89%** once the cohort was restricted to patients whose outcome window had fully elapsed. A label that moves this much under reasonable re-slicing is measuring an artefact, not a real quantity. The root cause: the definition conflated *genuine lapse* with *data-window truncation*, a patient's last visit being old simply because that is where their history ends in the extract.

**Attempt 3 - the final label.** Anchored to each patient's **second-to-last completed visit** (the prediction point), with the outcome observed from their *subsequent* activity. A patient is churned if they had no completed visit within their clinically-advised interval (`ReaMonths`) plus a 3-month grace. The cohort is bounded once, cleanly, to patients whose full outcome window closes within the data -  everyone else is explicitly excluded as unobservable, with counts logged.

This label is **stable (37.8% churn), leakage-free, and mirrors production scoring**: in a live system each patient would be evaluated against their own advised interval as their window matures, which is exactly what this label does.

*Why not a fixed observation date?* An observation-date proxy would eliminate the recency confound but discards patients seen after it and does not match how the model would actually score in deployment. The last-visit anchor is business-truthful; its one trade-off (a mild skew toward less-recent patients) is handled by bounding the observable cohort up front and stating it as a limitation.

### Features

Eight features, each computed **strictly from data at or before the prediction point** (the index visit) to prevent leakage:

- **Engagement:** visit count, tenure, average inter-visit gap
- **Value:** total dispense spend, order count (both as-of the index visit)
- **Demographics:** age (computed as-of the index visit, not today), sex
- **Clinical:** recall interval (`ReaMonths`)

Every feature was individually verified against a known floor or range before use. Undefined values (e.g. inter-visit gap for single-visit patients) are left as `NaN` and handled natively by the model rather than imputed.

### Modelling

A regularised XGBoost classifier, evaluated on a **temporal train/test split** (train on earlier prediction points, test on later ones). A random split would let the model learn from the future, inflating the score meaninglessly. AUC is used as the primary metric, being threshold-independent and robust to the mild train/test base-rate shift.

---

## Results

| Metric | Value |
|---|---|
| Final cohort | 5,029 patients (established, fully-observable outcome) |
| Churn rate | 37.8% |
| **Test AUC** | **0.63** |
| Train AUC | 0.71 (an 0.08 gap — controlled overfitting) |
| **Targeting lift vs random** | **1.38×** |
| **Recovered revenue (test slice, per cycle)** | **£660** |

**Strongest predictor:** recall interval (`ReaMonths`), by a wide margin, consistent with clinical expectation, as interval length reflects age and risk profile.

**Business value, in plain terms.** Contacting the top 20% of patients by model-predicted risk catches **77 real churners** versus **56** from contacting a random 20%, a 1.38× improvement on the same outreach budget. Valued at the practice's own average revenue per completed visit (£157, derived from £4.6M of order data across 29,517 completed visits) and a conservative 20% intervention success rate, the model adds **£660 of recovered revenue on the held-out test slice alone**, per recall cycle.

*All figures are revenue, not margin (no cost-of-goods data was available); margin would reduce the absolute £.*

---

## Limitations

Stated openly, because they shape how the result should be read:

- **Single-visit sparsity.** A majority of the practice's patients have only one completed visit in the data window, providing insufficient history for a churn prediction. The model is therefore scoped to *established* patients (2+ visits). Single-visit patients are a first-return / acquisition problem, out of scope here.
- **No contact-lens plan data.** This practice runs no contact-lens direct-debit scheme, so the strongest sector-standard retention signal is unavailable in the extract.
- **Recency skew.** The last-visit-anchored label mildly over-represents less-recently-seen patients (a consequence of requiring a fully-observable outcome). This is bounded and disclosed rather than designed away.
- **Modest AUC.** 0.63 reflects a genuine ceiling of the available features on this cohort; two independent model configurations land at the same test AUC, indicating the constraint is the data, not the tuning. Richer behavioural signals (appointment lead times, DNA history, a CL plan) would be the route to improvement.

---

## Tech stack

Python · pandas · scikit-learn · XGBoost · Jupyter · conda (native arm64)

## Repository structure

```
recall-engine/
├── docs/    
│   └──design.md             # priority score design(churn x value)
├── notebooks/
│   ├── 01_profiling.ipynb     # data → label → features → temporal split
│   └── 02_modelling.ipynb     # model → evaluation → business value
├── src/
│   ├── __init__.py
│   ├── pseudonymise.py        # raw → pseudonymised working data
│   ├── dataset.py             # load, clean, derive model inputs
│   ├── labels.py              # per-patient churn label (final anchor)
│   ├── features.py            # leakage-safe features up to the index visit
│   ├── model.py               # temporal split, XGBoost, AUC
│   └── pipeline.py            # single-command entry point
├── data_raw/                  # real exports (gitignored)
├── data_working/              # pseudonymised working data (gitignored)
├── models/                    # trained artifacts (gitignored)
└── outputs/                   # charts and result tables
```

## Reproducibility & next steps

The notebooks run top-to-bottom from a clean kernel, with all paths anchored to the project root and a fixed random seed. The label, feature, and modelling logic has since been refactored out of the notebooks into importable `src/` modules (data-loading, labelling, feature-engineering, and modelling as separate stages), runnable end-to-end from raw data to a trained model with a single command: `python -m src.pipeline`. Because the underlying data is real patient data, this repository documents the work rather than shipping the data itself. 

Development is now focused on productionising this work: a priority layer ranking patients by expected recovered revenue (churn probability × dispense value — see docs/design.md), deployment as a Dockerized FastAPI service on AWS with CI/CD and drift monitoring, and a synthetic data generator producing VisionPlus-shaped records so the full pipeline can be run end-to-end by anyone.

---

*Built as the flagship project of a data science portfolio. Data used with permission; all patient-identifying information removed prior to analysis.*