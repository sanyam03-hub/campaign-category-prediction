# Lessons learned

These are the problems found while building the original version of this system, and what this
codebase does about each.

## 1. Features and labels shared a time window (leakage)

The original features (`total_clicked`, `days_since_last_click`) were computed over the same
window that defined the label. Every labelled user had at least one click by construction, so the
most important feature partly encoded the label period, and scored users who never clicked sat at a
value the model had never seen in training.

**Here:** three consecutive windows. `config.get_windows` refuses to run if the feature window does
not end before the label window starts. `has_clicked` is an explicit flag.

## 2. Lifetime flags could see the future

Flags like `has_premium` were computed with a lower date bound but no upper bound, so training
rows could contain events that happened after the label window.

**Here:** every flag is bounded at the feature-window end (`features.build_app_features`), and a
test (`test_lifetime_flag_cannot_see_the_future`) covers it.

## 3. A headline hit rate hid a majority-class baseline

With one category making up a large share of clicks, "always predict the biggest category" scored
higher than the model. Dividing the model's hit rate by `1/number_of_classes` made it look several
times better than random, which was the wrong comparison.

**Here:** `validate` reports the hit rate beside the uniform baseline, the majority-class baseline,
whether the model beats it, and the hit rate with the majority class removed. Read the last number first.

## 4. A format change produced silent NULLs

The export format changed from JSON to CSV. The JSON parser returned NULL for every field instead of
failing, the rows were appended, and a month of data looked fine until validation.

**Here:** a suffix allow-list and a NULL-rate gate in `ingest`, and `test_csv_content_in_a_json_reader...`
reproduces the exact failure.

## 5. Appending data is not idempotent

Re-running a day's ingestion duplicated its rows.

**Here:** one file per day, replaced atomically.

## 6. The "no data" job could report success

Ingestion exited early when no files were found, and the scheduler counted that as a success, so the
next step scored on stale data.

**Here:** `predict` has a freshness gate that fails the run.

## 7. Cleanup that only removed part of the noise

Generic campaign names were removed only if they had landed in one particular category, so the rest
kept arbitrary categories. The daily job then re-added the removed names as "new".

**Here:** generics are removed before classification and recorded in `excluded_generic_campaigns.csv`,
and `classify_new_only` skips them.

## 8. Settings that did nothing

`is_unbalance=True` was set on a plain multiclass objective, where LightGBM ignores it. Documentation
claimed it handled class imbalance.

**Here:** it is not set. Class weights are explicit and opt-in.

## 9. Overrides that matched nothing

Manual overrides matched campaign names by exact string. Two keys differed from the real names by a
currency symbol and matched nothing, with no warning.

**Here:** matching uses normalised names, and unmatched override keys are printed.

## 10. Proof of business impact needs an experiment

Offline accuracy and hit rate do not show a click-through improvement. The honest next step is an
A/B test: send the predicted category to a treatment group and compare against a control.
