# Recall Engine
 
![CI](https://github.com/arjandhinsa/recall-engine/actions/workflows/ci.yml/badge.svg)
 
**Predicting which patients will fail to return for their recommended eye examination, using real practice data, then translating that prediction into recovered revenue.**
 
An end-to-end machine learning system built on ~38,000 patients' worth of real appointment, dispensing, and prescription data from an independent optometry practice. A temporally-validated churn model (AUC 0.63) is combined with patient dispense value to rank recalls by expected revenue per call, recovering **£3,183 per cycle versus £2,606 for churn-only ranking and £1,572 for random outreach** on held-out data. The service is deployed, monitored for drift, and redeploys itself on every push.
 
> The most substantial work in this project was not the model. It was diagnosing and rejecting **two structurally-flawed churn definitions** before arriving at a label that is business-truthful, leakage-free, and consistent with how the model would score in live deployment. That reasoning chain is documented below.
 
---
 
## Live demo
 
**https://recall.seyn.co.uk/docs**, the API running on AWS. The public demo runs entirely on synthetic data.
 
- `GET /recall-list?top_n=20`, the ranked recall list
- `POST /score`, churn probability, expected value and priority for one patient
- `GET /monitoring`, current data and prediction drift status
- `GET /monitoring/report`, full Evidently drift dashboard
- `GET /health`
### Run it locally
 
No real patient data is required or included:
 
```bash
docker compose up
```
 
Then open http://localhost:8000/docs.
 
To regenerate the synthetic practice and retrain from scratch:
 
```bash
python -m src.make_synthetic
DATA_DIR=data_working/synthetic python -m src.pipeline
DATA_DIR=data_working/synthetic python -m src.batch_score
```
 
---
 
## Problem
 
Independent optometry practices lose measurable revenue when patients drift away between recall cycles. A lapsed patient means a missed sight test *and* the associated dispense (glasses or contact lenses). Recalls are the practice's core retention mechanism, but outreach is limited: staff can only call so many patients. The question this project answers:
 
**Given a patient's history, who should the practice call first, and what is that call worth?**
 
If the model concentrates high-value churners into a small, contactable list, a fixed outreach budget recovers more revenue than contacting patients at random.
 
---
 
## Data
 
Four raw exports from the practice's VisionPlus management system:
 
| Table | Rows | Content |
|---|---|---|
| Patients | 38,379 | Demographics (pseudonymised) |
| Appointments | 39,040 | Every appointment, with attendance status |
| Orders | 127,222 | Dispensing lines, frames, lenses, coatings, with value |
| Prescriptions | 30,005 | Clinical data, including the recall interval (`ReaMonths`) and projected recall date |
 
**Data governance.** All patient-identifying fields (names, addresses, full postcodes, dates of birth) are removed or reduced before any analysis. Patient references are converted to non-reversible salted HMAC-SHA256 keys, with the salt held outside the repository. All raw and working data is excluded from version control; nothing patient-identifying is committed at any point, and nothing patient-identifying reaches the deployed service.
 
---
 
## Approach
 
### Building an honest churn label
 
Defining "churn" on a fixed historical extract is deceptively hard, and getting it wrong produces a model that scores well and means nothing. The label went through three iterations:
 
**Attempt 1, "overdue past the projected recall date."** This produced an implausible **84.5% churn rate**. Investigation revealed the recall date field was a *projected* next-due date (last visit + interval), not a recall the patient had responded to, so a fixed window after it systematically missed patients who returned early or late.
 
**Attempt 2, "gap since last visit exceeds the recall interval."** This gave **65%**, then swung to **89%** once the cohort was restricted to patients whose outcome window had fully elapsed. A label that moves this much under reasonable re-slicing is measuring an artefact, not a real quantity. The root cause: the definition conflated *genuine lapse* with *data-window truncation*, a patient's last visit being old simply because that is where their history ends in the extract.
 
**Attempt 3, the final label.** Anchored to each patient's **second-to-last completed visit** (the prediction point), with the outcome observed from their *subsequent* activity. A patient is churned if they had no completed visit within their clinically-advised interval (`ReaMonths`) plus a 3-month grace. The cohort is bounded once, cleanly, to patients whose full outcome window closes within the data; everyone else is explicitly excluded as unobservable, with counts logged.
 
This label is **stable (37.8% churn), leakage-free, and mirrors production scoring**: in a live system each patient would be evaluated against their own advised interval as their window matures, which is exactly what this label does.
 
*Why not a fixed observation date?* An observation-date proxy would eliminate the recency confound but discards patients seen after it and does not match how the model would actually score in deployment. The last-visit anchor is business-truthful; its one trade-off (a mild skew toward less-recent patients) is handled by bounding the observable cohort up front and stating it as a limitation.
 
### Features
 
Eight features, each computed **strictly from data at or before the prediction point** (the index visit) to prevent leakage:
 
- **Engagement:** visit count, tenure, average inter-visit gap
- **Value:** total dispense spend, order count (both as-of the index visit)
- **Demographics:** age (computed as-of the index visit, not today), sex
- **Clinical:** recall interval (`ReaMonths`)
Every feature was individually verified against a known floor or range before use. Undefined values (e.g. inter-visit gap for single-visit patients) are left as `NaN` and handled natively by the model rather than imputed.
 
A single `prepare_features` function encodes and orders features for training, live scoring, and the API, so encoding cannot silently diverge between them (training/serving skew).
 
### Modelling
 
A regularised XGBoost classifier, evaluated on a **temporal train/test split** (train on earlier prediction points, test on later ones). A random split would let the model learn from the future, inflating the score meaninglessly. AUC is used as the primary metric, being threshold-independent and robust to the mild train/test base-rate shift.
 
### From risk to priority
 
Churn probability alone is the wrong ranking. A 90%-risk patient worth £40 per visit is a worse call than a 60%-risk patient worth £600. The priority layer multiplies the two:
 
```
priority = P(churn) × expected dispense value per visit
```
 
Value is shrunk toward the practice mean, `(total_spend + k·£157) / (visits + k)`, so patients with several visits and no dispense are correctly pulled down (zero dispense across visits is behavioural evidence, not missing data) while a single visit is not over-interpreted. Value obeys the same leakage rule as the features, using only orders at or before the index visit. Full reasoning in [`docs/design.md`](docs/design.md).
 
---
 
## Results
 
| Metric | Value |
|---|---|
| Final cohort | 5,029 patients (established, fully-observable outcome) |
| Churn rate | 37.8% |
| **Test AUC** | **0.63** |
| Train AUC | 0.71 (an 0.08 gap, controlled overfitting) |
 
**Strongest predictor:** recall interval (`ReaMonths`), by a wide margin, consistent with clinical expectation, as interval length reflects age and risk profile.
 
### Targeting backtest
 
Top-20% outreach on the held-out test period, all strategies valued with per-patient dispense values and a conservative 20% intervention success rate:
 
| Strategy | £ recovered per cycle | Churners caught | Lift vs random |
|---|---|---|---|
| Random 20% | £1,572 | 56 (expected) | 1.0× |
| Churn-ranked | £2,606 | 77 | 1.66× |
| **Priority-ranked (churn × value)** | **£3,183** | 62 | **2.02×** |
 
Priority ranking deliberately trades away low-value churners (62 caught versus 77) for high-value ones, recovering **22% more revenue on the identical outreach budget**. In incremental terms, over random outreach, churn-ranking adds £1,034 per cycle and priority-ranking adds £1,611, a 56% improvement from the value layer alone.
 
*All figures are revenue, not margin (no cost-of-goods data was available); margin would reduce the absolute £. The 20% intervention rate is an assumption, currently being replaced by a measured one (see below).*
 
### Measuring the intervention effect
 
The 62 top-priority patients from a live scoring run were pair-matched on priority score and randomly split into 31 called and 31 not called, with the allocation frozen and the outcome pre-registered (booked an appointment within six weeks) before any contact was made. The difference in booking rates between the two groups will replace the assumed 20% success rate with a measured one.
 
---
 
## Deployment
 
| Layer | Implementation |
|---|---|
| Service | FastAPI, Dockerized, running on EC2 behind Caddy with automatic TLS on a custom domain |
| Registry | Amazon ECR |
| CI | GitHub Actions runs the test suite on every push; tests gate deployment |
| CD | On merge to main: regenerate synthetic data, retrain, build, push to ECR, redeploy via SSM, then verify the live endpoint |
| Auth | GitHub OIDC federation and an EC2 instance role, no long-lived credentials stored anywhere |
| Batch | `python -m src.batch_score` produces the ranked recall list the practice acts on |
 
Real and synthetic runs are separated by a `DATA_DIR` environment variable, which routes model artifacts and outputs into `real/` or `synthetic/` directories. The deployed image can only ever load synthetic artifacts.
 
## Monitoring
 
Every batch run compares the current scoring cohort against the training feature distribution saved alongside the model, using Evidently. It writes an HTML report and a machine-readable status file exposing per-feature drift, prediction drift, and an overall flag. A scheduled workflow reads the live `/monitoring` endpoint weekly and opens a GitHub issue if drift breaches threshold.
 
Retraining is not automatic, deliberately. Some of the measured drift is structural rather than real: the training cohort is anchored at the second-to-last visit and the scoring cohort at the last visit, so visit counts and tenure differ by construction. The useful signal is the trend across runs, not any single value, which is a judgement a human should make. Retraining on real practice data runs locally; it never leaves the practice.
 
---
 
## Limitations
 
Stated openly, because they shape how the result should be read:
 
- **Single-visit sparsity.** A majority of the practice's patients have only one completed visit in the data window, providing insufficient history for a churn prediction. The model is therefore scoped to *established* patients (2+ visits). Single-visit patients are a first-return / acquisition problem, out of scope here.
- **No contact-lens plan data.** This practice runs no contact-lens direct-debit scheme, so the strongest sector-standard retention signal is unavailable in the extract.
- **Recency skew.** The last-visit-anchored label mildly over-represents less-recently-seen patients (a consequence of requiring a fully-observable outcome). This is bounded and disclosed rather than designed away.
- **Modest AUC.** 0.63 reflects a genuine ceiling of the available features on this cohort; two independent model configurations land at the same test AUC, indicating the constraint is the data, not the tuning. Richer behavioural signals (appointment lead times, DNA history, a CL plan) would be the route to improvement.
- **Synthetic demo scores higher than reality.** The public demo reaches AUC 0.72 because generated data is cleaner and driven by one latent trait. Real behaviour is messier and partly unpredictable; 0.63 is the honest number.
---
 
## Tech stack
 
Python · pandas · scikit-learn · XGBoost · FastAPI · Docker · AWS (EC2, ECR, IAM, SSM) · GitHub Actions · Evidently · pytest
 
## Repository structure
 
```
recall-engine/
├── .github/workflows/
│   ├── ci.yml                 # test, build, push to ECR, redeploy, verify
│   └── drift-check.yml        # weekly drift check, opens an issue on breach
├── docs/
│   └── design.md              # priority score design (churn × value)
├── notebooks/
│   ├── 01_profiling.ipynb     # data → label → features → temporal split
│   ├── 02_modelling.ipynb     # model → evaluation → business value
│   └── 03_priority.ipynb      # priority backtest vs churn-only and random
├── src/
│   ├── pseudonymise.py        # raw → pseudonymised working data
│   ├── make_synthetic.py      # fake VisionPlus-shaped practice for the demo
│   ├── dataset.py             # load, clean, derive model inputs
│   ├── labels.py              # per-patient churn label (final anchor)
│   ├── features.py            # leakage-safe features up to the index visit
│   ├── value.py               # shrinkage-smoothed dispense value, tiers
│   ├── priority.py            # ranked expected-revenue recall list
│   ├── cohort.py              # live scoring cohort (anchored at last visit)
│   ├── model.py               # temporal split, XGBoost, shared feature prep
│   ├── monitoring.py          # Evidently drift report and status
│   ├── pipeline.py            # train end-to-end, save model and reference
│   ├── batch_score.py         # score everyone, write the recall list
│   └── api.py                 # FastAPI service
├── tests/                     # pure-function tests with hand-checked fixtures
├── Dockerfile
├── docker-compose.yml
├── data_raw/                  # real exports (gitignored)
├── data_working/              # pseudonymised working data (gitignored)
├── models/{real,synthetic}/   # model + training reference (gitignored)
└── outputs/{real,synthetic}/  # recall lists and drift reports (gitignored)
```
 
## Reproducibility
 
The pipeline runs end-to-end from a single command with a fixed random seed, and the synthetic generator means anyone can reproduce the full system without access to the practice's data:
 
```bash
python -m src.make_synthetic
DATA_DIR=data_working/synthetic python -m src.pipeline
DATA_DIR=data_working/synthetic python -m src.batch_score
```
 
Because the underlying data is real patient data, this repository documents the work rather than shipping the data itself.
 
---
 
*Data used with permission; all patient-identifying information removed prior to analysis.*