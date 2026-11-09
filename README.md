# Instamart Customer Intelligence & Support Copilot

## Overview

This repository implements an end-to-end, reproducible customer intelligence and support copilot over **synthetic fictional data**. It connects four parts: (A) raw-data audit, cleaning, SQLite and SQL analysis; (B) customer segmentation and lapse prediction; (C) privacy-safe support-ticket triage; and (D) retrieval-grounded policy answering and a tool-using refund agent. All monetary values are in Indian Rupees (INR). No real customer data, external model API, API key or network service is used.

The two supplied starter files, `generate_data.py` and `mock_llm.py`, are included at the repository root. `mock_llm.py` is an offline deterministic mock and is used for all language-model interactions.

## Reproduce

Run these commands from the repository root. Python 3.11+ is required. On Windows PowerShell use the activation command shown; on macOS/Linux use the alternative.

1. `python -m venv .venv`
2. Windows PowerShell: `.venv\Scripts\Activate.ps1` (macOS/Linux: `source .venv/bin/activate`)
3. `python -m pip install -r requirements.txt`
4. `python generate_data.py`
5. `python part_a.py`
6. `python validate_data.py`
7. `python run_sql.py`
8. `python pandas_analysis.py`
9. `python part_b.py`
10. `python part_c.py`
11. `python part_d.py`

The generator is deterministic (seed 42) and prints `generated: customers=1200 products=40 orders_raw=13084 tickets=164`. All scripts write the required data and outputs under `data/`, `outputs/` and `state/`. The generated database and model artifacts are reproducible from the source files and the fixed random seeds.

## Part A — Data Foundation: Clean, Store, Query

### Raw audit

| Raw table | Shape |
|---|---:|
| customers | 1,200 × 5 |
| products | 40 × 4 |
| orders | 13,084 × 9 |
| tickets | 164 × 5 |

Raw column data types:

| Column | pandas dtype |
|---|---|
| `customers.customer_id` | `object` |
| `customers.city` | `object` |
| `customers.signup_date` | `object` |
| `customers.age` | `string` |
| `customers.gender` | `object` |
| `products.product_id` | `object` |
| `products.product_name` | `object` |
| `products.category` | `object` |
| `products.price_inr` | `int64` |
| `orders.order_id` | `object` |
| `orders.customer_id` | `object` |
| `orders.product_id` | `object` |
| `orders.quantity` | `int64` |
| `orders.order_ts` | `object` |
| `orders.status` | `object` |
| `orders.discount_pct` | `int64` |
| `orders.order_amount` | `string` |
| `orders.delivery_minutes` | `string` |
| `tickets.ticket_id` | `object` |
| `tickets.customer_id` | `object` |
| `tickets.order_id` | `object` |
| `tickets.text` | `object` |
| `tickets.true_category` | `object` |

Missing values by raw column (tickets are read with `keep_default_na=False`, so blank ticket text is counted separately rather than as pandas NA):

| Column | Missing count | Missing % |
|---|---:|---:|
| `customers.customer_id` | 0 | 0.00% |
| `customers.city` | 0 | 0.00% |
| `customers.signup_date` | 0 | 0.00% |
| `customers.age` | 61 | 5.08% |
| `customers.gender` | 30 | 2.50% |
| `products.product_id` | 0 | 0.00% |
| `products.product_name` | 0 | 0.00% |
| `products.category` | 0 | 0.00% |
| `products.price_inr` | 0 | 0.00% |
| `orders.order_id` | 0 | 0.00% |
| `orders.customer_id` | 0 | 0.00% |
| `orders.product_id` | 0 | 0.00% |
| `orders.quantity` | 0 | 0.00% |
| `orders.order_ts` | 0 | 0.00% |
| `orders.status` | 0 | 0.00% |
| `orders.discount_pct` | 0 | 0.00% |
| `orders.order_amount` | 53 | 0.41% |
| `orders.delivery_minutes` | 2541 | 19.42% |
| `tickets.ticket_id` | 0 | 0.00% |
| `tickets.customer_id` | 0 | 0.00% |
| `tickets.order_id` | 0 | 0.00% |
| `tickets.text` | 0 | 0.00% |
| `tickets.true_category` | 0 | 0.00% |

Specific raw-data audit checkpoints:

| Check | Observed |
|---|---:|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Pending rows | 656 |
| Distinct order IDs | 12,309 |
| Unparseable order_amount values (numeric coercion) | 120 |
| Negative delivery times | 106 |
| Blank delivery times | 2541 |
| Blank customer ages | 61 |
| Customer ages outside 10–100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct nonblank raw gender spellings | 10 |
| Blank genders | 30 |
| Blank ticket text rows | 4 |

### Cleaning rules and results

- **Customers:** city values were stripped and title-cased, collapsing 24 raw spellings to six canonical cities. Ten raw nonblank gender spellings were mapped to `male`, `female` or `unknown`; all 30 blank genders became `unknown`. The 61 blank ages and six out-of-range ages (67 values total) were set missing and filled using the city median via `groupby('city').transform('median')`. `signup_date` was parsed as a datetime.
- **Orders:** `order_ts` was parsed; 119 exact duplicates were removed; then one latest timestamp row was retained per `order_id`, removing 656 earlier pending/repeated-order rows from the post-dedup set (the cleaned table contains one row per order). Amounts were converted to numeric and 112 missing/unparseable values remaining after duplicate/order-version resolution were imputed from product catalogue price × quantity × (1 − discount_pct/100), rounded to two decimals. The raw export had 106 negative delivery durations and 2,541 blank durations; negative values were set missing. 774 missing completed-order delivery durations were filled with the completed-order median for the product category using `groupby('category').transform('median')`; cancelled orders retained missing delivery time.
- **Products:** the 40 catalogue rows were retained; `price_inr` is numeric.

Clean customer city counts: `Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184`. Gender counts: `female 610, male 560, unknown 30`. There are zero missing cleaned ages and the cleaned ages range from 18 to 60.

Clean order table: **12,309 rows**, **11,110 completed**, **1,199 cancelled**, zero pending. There are zero missing order amounts, zero missing delivery times among completed orders, and 1,199 missing delivery times among cancelled orders. Completed-order revenue is **INR 2,998,668.80**. Clean CSVs are `data/clean/customers.csv`, `data/clean/products.csv`, `data/clean/orders.csv`; SQLite database `data/instamart.db` contains `customers`, `products`, `orders`.

### SQL analysis

Seven queries are stored in `sql/queries.sql` and executed using `pandas.read_sql_query` against SQLite.

| Query | Technique / result |
|---|---|
| Q1 | WHERE then HAVING: 30 customers meet the qualifying spend rule. |
| Q2 | Three-table join: 30 city/category rows; total revenue INR 2,998,668.80; largest cell is Bengaluru / Electronics at INR 233,909.45. |
| Q3 | Scalar subquery: 11 products priced above the average catalogue price. |
| Q4 | LEFT JOIN: 25 customers have no completed order. |
| Q5 | CTE: 550 customers have spend above the average customer spend (INR 2,552.06). |
| Q6 | Ranking windows: 1175 customer/city rows; 8 rows have RANK() different from ROW_NUMBER(). Ties share a RANK position, while ROW_NUMBER assigns a unique sequence, so ranks can skip. |
| Q7 | Running total: 183 daily rows; final cumulative revenue INR 2,998,668.80. |

Top completed-spend customer in each city:

| City | Customer | Spend (INR) |
|---|---|---:|
| Bengaluru | C0082 | 12,856.20 |
| Chennai | C0930 | 10,151.40 |
| Delhi | C0291 | 13,742.60 |
| Hyderabad | C0743 | 12,730.05 |
| Mumbai | C0693 | 14,766.55 |
| Pune | C0426 | 10,377.10 |

### Pandas cross-check and exploration

The pandas pivot table with margins reproduces the SQL Q2 grand total of **INR 2,998,668.80**. Bengaluru completed-order revenue is INR 523,643.40 (mean INR 276.47; 1,894 orders); Mumbai revenue is INR 479,058.50 (mean INR 297.00; 1,613 orders).

Completed order amount: mean INR 269.91, median INR 135.00, skewness 4.61; Q1 INR 64.80, Q3 INR 297.00, IQR INR 232.20; lower fence INR -283.50, upper fence INR 645.30; outliers below/above fences: 0/897. The distribution is strongly right-skewed: a relatively small number of large baskets raise the mean above the median. The largest absolute correlation is order_amount with discount_pct (-0.095); order_amount with quantity is 0.087. This correlation is small and does not establish causation.

## Part B — Customer Segments and Lapse Prediction

### RFM and segmentation

Whole-window RFM uses completed orders as of 2025-10-01 00:00:00: 1,175 customers; recency median/min/max 19/0/180 days; frequency mean/max 9.46/33; monetary mean/max INR 2,552.06/INR 14,766.55.

StandardScaler was fitted to the three RFM columns. KMeans used `n_init=10`, `random_state=42` for K=2..8:

| K | Inertia | Silhouette |
|---:|---:|---:|
| 2 | 1820.76 | 0.465 |
| 3 | 1268.23 | 0.445 |
| 4 | 743.07 | 0.526 |
| 5 | 617.37 | 0.480 |
| 6 | 503.17 | 0.482 |
| 7 | 408.46 | 0.416 |
| 8 | 366.50 | 0.412 |

K=4 was selected: it has the strongest silhouette in the tested range and an interpretable four-segment profile. Inertia always falls as K rises because more clusters can fit the training points more closely; the lowest inertia at K=8 is not, by itself, evidence that eight business segments are useful. The elbow, silhouette and actionability must be considered together.

| Cluster ID | Business segment | Customers | Mean recency (days) | Mean frequency | Mean monetary (INR) | Premium |
|---:|---|---:|---:|---:|---:|---|
| 0 | Occasional Customers | 436 | 28.86 | 3.77 | 1057.62 | No |
| 1 | Premium Customers | 138 | 24.12 | 6.77 | 7129.70 | Yes |
| 2 | Loyal Customers | 419 | 8.77 | 19.40 | 3410.02 | No |
| 3 | At-Risk Customers | 182 | 109.04 | 2.20 | 685.99 | No |

Cluster IDs are arbitrary labels, not an ordinal ranking. Segment actions: Premium Customers — proactive high-value service and tailored benefits; Loyal Customers — loyalty rewards and cross-category engagement; Occasional Customers — relevant reminders and basket-building offers; At-Risk Customers — win-back outreach and service-recovery checks. PCA on the scaled RFM matrix explains 0.634 and 0.218 of variance (total 0.852).

### Lapse definition, leakage controls and model results

Cutoff: **2025-09-01**. Cohort: every customer with at least one completed order before the cutoff. `lapsed=1` when there is no completed order from 2025-09-01 through 2025-09-30; otherwise `lapsed=0`. Cohort size 1,156; lapsed 443 (38.32%); active 713. Split: 924 training and 232 test rows with `test_size=0.2`, `stratify=y`, `random_state=42`.

Features use only pre-cutoff orders plus the clean customer table: recency, frequency, monetary, average discount, cancellation rate over final pre-cutoff orders, average delivery time, most frequent product category, city, age, gender and tenure. No order on or after 2025-09-01 enters the feature matrix, and whole-window segment labels are not model features. Missing feature values are asserted absent. OneHotEncoder (`drop='first'`) and StandardScaler are inside a scikit-learn Pipeline and fitted only on training rows (and within each CV fold); test rows are transformed only. The learned training-scaler mean for frequency is 8.115, compared with full-cohort frequency mean 8.029; they differ because the scaler is fitted on the training subset, not the entire cohort.

Baseline always predicting not-lapsed: accuracy 61.6%, lapsed-class F1 0.00. GridSearchCV used 5-fold CV and F1 scoring:

| Model | Search grid | Best parameters | Best CV F1 |
|---|---|---|---:|
| Logistic Regression (`max_iter=1000`) | C ∈ {0.01, 0.1, 1, 10, 100} | C=100 | 0.651 |
| Random Forest (`n_estimators=100`, `random_state=42`) | max_depth ∈ {3,5,8}; min_samples_leaf ∈ {5,10,20} (45 fits) | depth=8, leaf=10 | 0.686 |

Untouched test-set metrics below are for the **positive/lapsed class** (confusion matrices use `[[TN, FP], [FN, TP]]`):

| Model | Confusion matrix | Precision | Recall | F1 | Accuracy |
|---|---|---:|---:|---:|---:|
| Logistic Regression | `[[111, 32], [31, 58]]` | 0.644 | 0.652 | 0.648 | 0.728 |
| Random Forest | `[[111, 32], [32, 57]]` | 0.640 | 0.640 | 0.640 | 0.724 |

Logistic Regression was selected because its test F1 is marginally higher in this run. Missing a lapsed customer means losing an opportunity to intervene, while wrongly targeting an active loyal customer wastes a discount and may train the customer to wait for promotions. The two errors therefore have different business costs; campaign thresholds should be chosen with retention capacity and offer cost in mind.

### Class-balance experiment and interpretation

The positive class was oversampled **only in the training set**, retaining all original training rows and drawing lapsed rows with replacement until both classes had 570 rows. On the unchanged test set, balanced Logistic Regression achieved precision 0.589, recall 0.854, F1 0.697, compared with the unbalanced Logistic Regression above. Recall moved most (from 0.652 to 0.854); this finds more potential lapsers but creates more false-positive retention contacts, and precision did not improve.

Five highest Random Forest feature importances:

| Feature | Importance |
|---|---:|
| monetary | 0.280 |
| frequency | 0.278 |
| recency | 0.109 |
| avg_delivery_minutes | 0.060 |
| cancellation_rate | 0.052 |

Three Logistic Regression coefficient signs:
- **recency**: positive coefficient (0.039). Higher recency (longer time since last order) is associated with higher lapse odds.
- **frequency**: negative coefficient (-1.698). Higher order frequency is associated with lower lapse odds.
- **monetary**: positive coefficient (0.284). Higher historical spend is associated with higher lapse odds in this fitted model; this is an association, not a causal effect.

The selected fitted pipeline is saved to `outputs/lapse_model.joblib`, reloaded with `joblib.load`, and its test predictions match the original predictions exactly. `outputs/customer_segments.csv` contains `customer_id, recency, frequency, monetary, segment_id, segment_name, is_premium`. `outputs/retention_list.csv` contains the top 10 Premium-segment test customers by predicted lapse probability with `customer_id, lapse_probability, actually_lapsed`; 3 of these 10 customers actually lapsed. **This retention list is illustrative and not a leakage-free evaluation because the Premium flag is based on whole-window RFM, which includes the September label window.**

## Part C — Support Ticket Triage with Guardrails

### Guardrails and privacy

The pipeline processes all 164 tickets in source order. Four empty/whitespace tickets are marked `skipped_empty`; four tickets above `MAX_INPUT_TOKENS=400` are marked `flagged_long` (their estimated input length is 734 tokens). Before model prompts, the pipeline applies the specified regular expressions in order: email → `[EMAIL]`, phone → `[PHONE]`, order ID → `[ORDER_ID]`. Redactions: 55 emails, 91 phone numbers, 132 order IDs; total 278; rechecking redacted prompts finds zero remaining matches. The user message contains exactly the redacted ticket text and no additional wrapper.

Zero-shot and few-shot prompts both require JSON-only output with one of four categories: billing, delivery, technical or fraud. Each call is wrapped in up to three retries with exponential backoff and jitter (`BASE_DELAY=0.05`); the mock's 19 transient failures per full run were all recovered. Output validation accepts only allowed JSON categories or exactly one allowed category word extracted with `re.findall(r'[a-z]+', ...)`; other output is `flagged_invalid_output` and sent to human review.

### Zero-shot vs few-shot

| Measure | Zero-shot | Few-shot |
|---|---:|---:|
| Classified tickets | 147 | 147 |
| Accuracy on classified tickets | 85.7% | 100.0% |
| Total tokens | 11,584 | 18,604 |
| Tokens per classified ticket | 78.80 | 126.56 |

Zero-shot confusion matrix (rows are actual categories; columns are predicted categories in the order billing, delivery, technical, fraud):

| Actual \ Predicted | billing | delivery | technical | fraud |
|---|---:|---:|---:|---:|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| technical | 10 | 0 | 29 | 0 |
| fraud | 11 | 0 | 0 | 23 |

The few-shot prompt used 47.76 additional tokens per classified ticket (7,020 total) and raised accuracy from 85.7% to 100.0%. In this deterministic mock, the added examples corrected fraud and technical tickets that zero-shot confused with billing. The additional prompt cost is justified when category errors can misroute security incidents; a production system should still monitor real token cost and validation failures.

Full-run status counts for both prompts: 135 `classified_json`, 12 `classified_keyword`, 9 `flagged_invalid_output`, 4 `flagged_long`, 4 `skipped_empty`. The zero-shot budget demonstration used TOKEN_BUDGET=6000 and ended with 75 `budget_exhausted` tickets (75 in this run). Final few-shot run used TOKEN_BUDGET=100000 and produced 164 rows in `outputs/triage_results.csv`; `outputs/triage_budget_demo.csv` contains the budget-stop demonstration.

Final routing counts: support_queue 102, security_team 34, human_review 17, premium_desk 11. Invalid, empty, long or budget-exhausted tickets go to `human_review` (NORMAL); fraud goes to `security_team` (HIGH); otherwise Premium customers go to `premium_desk` (HIGH), and remaining tickets go to `support_queue` (NORMAL). `outputs/triage_results.csv` columns are `ticket_id, customer_id, status, category, segment_name, priority, route, tokens_used`.

## Part D — Policy RAG Assistant and Refund Agent

### Chunking, retrieval and grounded answers

Five policy files are split into 40-word chunks with 10-word overlap (step size 30), producing 15 chunks: refund 4, delivery 3, payment 3, privacy 2, escalation 3. `TfidfVectorizer()` fits the chunk texts; queries are transformed with the same vectorizer and ranked by cosine similarity. For TF-IDF, term frequency measures how often a term occurs in a document, while inverse document frequency downweights terms common across documents. A limitation is that semantically equivalent phrases using different words may have low lexical similarity.

With overlap: answerable retrieval **12/12**, expected-fact substring **12/12**, unanswerable fallback **2/2**. The top three chunks for `refund policy late delivery` are all from `refund_policy.txt`. With zero overlap (40-word chunks, step size 40): 13 chunks, retrieval 12/12, expected facts 10/12, unanswerable fallback 2/2. The questions that failed the expected-fact check without overlap were: What delivery time makes a completed order eligible for a full refund?, How soon is money returned when a payment fails but is deducted?. Overlap preserves sentences split at chunk boundaries, improving answer grounding.

`answer(question)` retrieves three chunks and sends the exact `CONTEXT:` / `[source]` / `QUESTION:` user-message structure to the offline `POLICY ASSISTANT` mock. If no context sentence matches, the answer begins `I could not find`.

### Refund tools and ReAct agent

Five tools are implemented and mapped in `TOOL_MAP`: `get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`. The agent reads orders from `data/instamart.db`, reuses policy retrieval, and appends issue/escalation actions to `outputs/refund_ledger.jsonl` with timestamps. Rules: only completed orders are eligible; delivery over 45 minutes yields a full order-amount refund; delivery in 45 minutes or less yields INR 0; amounts above INR 1000 are escalated and not automatically issued. All scenario requests are assumed to be within the policy's 7-day window.

| Scenario | Status | Refund (INR) | Model calls | Tokens |
|---|---|---:|---:|---:|
| S1 | REFUND | 128.25 | 5 | 952 |
| S2 | REFUND | 135.00 | 5 | 952 |
| S3 | ESCALATE | 1999.00 | 5 | 939 |
| S4 | DENIED | 0.00 | 4 | 649 |
| S5 | DENIED | 0.00 | 3 | 370 |
| S6 | NOT_FOUND | 0.00 | 2 | 94 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3066 |

S1 completes in five model calls. S7 makes six tool-call iterations and ends `FALLBACK_MAX_ITERATIONS`. Stop-safely demonstrations: S1 with token_limit=200 → `FALLBACK_TOKEN_LIMIT`; S1 with time_limit_seconds=0 → `FALLBACK_TIMEOUT`. Both return a human-handoff summary and preserve completed steps.

Crash-and-resume demonstration: the S1 run was deliberately interrupted after the second tool observation had been checkpointed. `resume_agent(run_id)` continued from the saved message list. Tools executed across the interrupted and resumed run, in order: **get_order, search_policy, calculate_refund, issue_refund**. No tool was repeated; final status `REFUND`, refund INR 128.25. Checkpoint JSON files are stored under `state/`. `outputs/agent_runs.jsonl` contains one JSON line for each of the seven scenarios; `outputs/refund_ledger.jsonl` contains simulated issue/escalation entries.

### ReAct vs plan-and-execute (under 250 words)

The measured ReAct runs used five model calls for S1, three for S5 and two for S6. A plan-and-execute design could ask for a complete sequence once and use fewer calls on a predictable happy path, reducing prompt tokens. However, refund execution depends on observations: an order may be missing, cancelled, delivered within the threshold, or require escalation above INR 1000. ReAct observes each tool result before choosing the next action, so it can stop safely on NOT_FOUND or DENIED and does not issue a refund before checking the order and policy. A one-shot plan risks following stale assumptions if an observation differs from the plan; replanning can recover but adds calls. For this workflow, ReAct is preferred because correctness, authorization and safe handoff matter more than minimising a few local mock calls.

## Limitations and Next Steps

- All data, policy documents, tickets and refund scenarios are synthetic and fictional; results do not describe real Instamart customers or policy.
- `mock_llm.py` is deterministic and offline. Its behaviour is useful for reproducible guardrail testing but does not measure real-world language-model quality.
- The retention list is illustrative and not a leakage-free evaluation because the Premium segment flag is derived from whole-window RFM that includes the September label window. The lapse-model features themselves use only pre-cutoff orders, and segment labels are not model inputs.
- Risk thresholds and retention actions are rule-based starting points. They should be calibrated with business costs and tested with controlled retention experiments before production use.
- Refund ledger entries are simulated records only; no payment is initiated. Human approval remains required for refunds above INR 1000.
- Before production, add access control, audit retention, monitoring, fairness review, real-data governance and integration tests. Keep personal data out of model prompts and secrets out of version control.
