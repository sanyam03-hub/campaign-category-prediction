# Design decisions

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Model | LightGBM, multiclass | Logistic regression, neural net | Tabular counts and flags, mixed scales, missing values, fast to train, gain-based importance. Not benchmarked against the alternatives. |
| Campaign labels | Keyword cascade, then optional sentence embeddings | LLM API | No per-call cost, no data leaving the machine. Rules are transparent and auditable. |
| Low-confidence semantic matches | Written to a review file | Force into a default category | A forced default quietly pollutes the biggest "catch-all" class. |
| Generic campaign names (`PN1`, `SMS2`) | Dropped before classification | Classify then filter | They carry no information, so any category assigned to them is arbitrary. |
| Daily refresh | Recompute features over a rolling window | Incremental state table | No drift and trivial recovery. Costs more compute per run. Switch only if runtime or cost forces it. |
| Label | Most-clicked category in the label window | Click share, multi-label | Matches how results are evaluated. Loses users who click several categories. |
| Output | Top-2 categories with scores | Full probability vector | Small and easy to filter in SQL. Scores are uncalibrated. |
| Storage | Parquet/CSV files, config-driven paths | A specific warehouse | Runs anywhere. The same logic ports to Spark SQL. |

## What is deliberately not here

- **Class re-weighting by default.** In the original project, blanket inverse-frequency weighting
  hurt accuracy. `model.class_weights` lets you try per-class weights deliberately, one change at a time.
- **A probability threshold for "send / don't send".** That needs a calibration step and an
  A/B test. See `lessons-learned.md`.
