# Campaign Category Prediction

Predict which **notification campaign category** each user is most likely to click, so a team can
target messages instead of broadcasting every campaign to everyone.

> **Status:** code and tests only. This repository contains no data and no trained model, and
> reports no results. Run it on your own dataset (see [Quick start](#quick-start)).

## What it does

1. **Classify campaigns** into categories using ordered keyword rules, with an optional
   sentence-embedding fallback for names the rules miss. Uninformative names (`PN1`, `SMS2`) are
   dropped, and low-confidence matches go to a review file instead of being forced into a category.
2. **Build labels**: each user's most-clicked category in a *label window*.
3. **Build features** from an earlier *feature window*: app behaviour, notification engagement, and
   how many messages of each category the user was sent.
4. **Train** a multiclass LightGBM model and report accuracy next to the uniform and
   majority-class baselines.
5. **Validate forward** on a later window, again next to the baselines.
6. **Score daily** over a rolling window, with a freshness check, a feature-order check, and a
   distribution-shift check before publishing.

## Why this repo is built the way it is

It started from a production system and is a cleaned rebuild that fixes the problems found there:
overlapping feature and label windows, a headline metric that a majority-class guess beat,
silent NULL rows after an export format change, non-idempotent ingestion, and manual overrides that
matched nothing. Each fix has a test. See [`docs/lessons-learned.md`](docs/lessons-learned.md).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # makes `python -m ccp` available
python -m pytest            # unit tests; need no data

cp config/pipeline.example.yaml config/pipeline.yaml
# 1. put your tables in data/  (see docs/data-contract.md)
# 2. edit config/pipeline.yaml: paths, event names, the three date windows
# 3. edit config/features.example.yaml so event names match your app events

python -m ccp classify  --config config/pipeline.yaml
python -m ccp labels    --config config/pipeline.yaml
python -m ccp train     --config config/pipeline.yaml
python -m ccp validate  --config config/pipeline.yaml
python -m ccp predict   --config config/pipeline.yaml --run-date 2024-05-15
```

Optional semantic fallback: `pip install -r requirements-optional.txt`, then set
`classification.use_semantic: true`.

## Project layout

```
config/    pipeline, taxonomy, feature and override examples (copy and edit)
src/ccp/   config, classify, labels, features, train, predict, validate, ingest, cli
tests/     unit tests built from tiny hand-written tables
docs/      architecture, data contract, design decisions, lessons learned
scripts/   scan_repo.sh: run before every push
data/      empty; your dataset goes here (git-ignored)
```

## Reading the results

- Read **`hit_rate_excluding_majority_class`** first. A model can beat random and still lose to
  "always predict the biggest category".
- `confidence` is the model's raw score, not a calibrated probability.
- Offline metrics do not prove a lift in click-through. That needs an A/B test.

## Limitations

- Keyword rules are examples; replace them with rules for your own campaign names.
- The label is one category per user. People who click several categories lose information.
- Rolling-window recomputation scans all events each run; fine for moderate data, not for very large tables.
- No model registry, scheduler, or CI is included.

## Before you push to GitHub

```bash
./scripts/scan_repo.sh     # must print "OK"
```

Only publish this if you have the right to. Check your employer's policy first.
