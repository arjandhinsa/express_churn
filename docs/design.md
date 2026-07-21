# Priority = P(churn) × value = expected £ recovered per call

Patients are ranked by a single number: their probability of churning multiplied by their expected dispense value per visit. This is the expected revenue recovered by one successful recall call.

Ranking by churn risk alone gets this wrong. A 90%-risk patient at £40/visit scores 0.9 × 40 = £36; a 60%-risk patient at £600/visit scores 0.6 × 600 = £360.
Churn-ranking calls the first; expected-value ranking calls the second and recovers ~10× the revenue on the same call.

Operationally, we can decide to contact the highest expected-value-at-risk patients first, which can replace
the blanket recall texts with a short, ranked call list.
This allows our fixed outreach budget to recover the most revenue.


### Value Definition + Shrinkage

**(total_spend + k×157) / (completed_visits + k)**
where k = 1

*£157 is the practice's average dispense revenue per completed visit; k is the weight of that prior, in phantom average visits*

*k=1 means one visit's worth of benefit-of-the-doubt*

Zero-dispense history is informative missingness: it is behavioural evidence, not missing data, so value is dragged down as visits accumulate without purchases.

Priority ranking covers only the model's cohort (2+ completed visits, valid ReaMonths). Registered patients who have never attended are not ranked low — they cannot be scored at all: no visits means no prediction point, no features, and no recall issued. They are out of scope.


### Tiers

The tiers will be split into £ boundaries, e.g., "Platinum = top 25% spend-per-visit". The tiers should be from the train-period only, because we want to avoid the test set influencing boundaries (so the backtest stays honest). 

The quartiles are: bronze, silver, gold, platinum.

So: compute cutoffs on training-period data, freeze them, apply to everyone.


### Leakage Rule For Values

Value is computed only from orders before the patient's index visit, mirroring the features rule. If post-prediction spend leaked in, the backtested £-lift would be ranked with future knowledge and the headline number would be inflated. In live deployment this is a non-issue, because "all orders so far" is all that exists; the rule protects the backtest.


### Deliverable

The deliverable is a ranked table, consisting of: a patient key, a churn probability, value, tier, priority, and cumulative expected revenue. 

