"""Command-line entry point:  python -m ccp <command> --config config/pipeline.yaml

Commands (run in this order the first time):
  classify   build the campaign -> category mapping
  labels     build training labels from the label window
  train      build features, train the model, write metrics
  validate   score the forward window and compare with baselines
  predict    score all users for a given date (daily mode)
  ingest     parse one raw JSON-lines file into the events folder
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from ccp import classify as C
from ccp import features as F
from ccp.config import get_windows, load_config
from ccp.ingest import freshness_gate, ingest_day, write_day_idempotent
from ccp.labels import build_labels
from ccp.predict import distribution_shift, load_model, predict_all
from ccp.train import save_model, train_model
from ccp.utils import CCPError, read_dir, read_table, write_atomic
from ccp.validate import forward_validation


def _write_json(obj: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, default=str)


def _taxonomy(cfg):
    return C.load_taxonomy(cfg["classification"]["taxonomy_file"])


def cmd_classify(cfg, args):
    tax = _taxonomy(cfg)
    ccfg = cfg["classification"]
    campaigns = read_table(cfg["paths"]["campaigns"])
    res = C.classify_campaigns(
        campaigns, tax, C.load_overrides(ccfg.get("overrides_file")),
        use_semantic=ccfg["use_semantic"], semantic_model=ccfg["semantic_model"],
        min_confidence=ccfg["semantic_min_confidence"],
    )
    C.check_mapping(res["mapping"], tax.categories)
    write_atomic(res["mapping"], cfg["output"]["mapping_file"])
    write_atomic(res["review"], cfg["output"]["review_file"])
    write_atomic(res["dropped"], str(Path(cfg["paths"]["artifacts_dir"]) / "excluded_generic_campaigns.csv"))
    print(f"mapped={len(res['mapping'])} review={len(res['review'])} dropped_generic={len(res['dropped'])}")
    if res["unmatched_overrides"]:
        print(f"WARNING: override keys that matched no campaign: {res['unmatched_overrides']}")


def cmd_labels(cfg, args):
    w = get_windows(cfg)
    tax = _taxonomy(cfg)
    events = read_dir(cfg["paths"]["notification_events_dir"])
    mapping = read_table(cfg["output"]["mapping_file"])
    ne = cfg["notification_events"]
    labels = build_labels(events, mapping, tax.categories, ne["sent"], ne["clicked"], w.label_start, w.label_end)
    write_atomic(labels, cfg["output"]["labels_file"])
    print(labels["dominant_category"].value_counts().to_string())


def _feature_inputs(cfg):
    tax = _taxonomy(cfg)
    spec = F.load_feature_spec(cfg["features"]["mapping_file"])
    app = read_table(cfg["paths"]["app_events"])
    events = read_dir(cfg["paths"]["notification_events_dir"])
    mapping = read_table(cfg["output"]["mapping_file"])
    prefix = cfg.get("user_id_prefix", "")
    if prefix:
        app = app[app["user_id"].astype(str).str.startswith(prefix)]
        events = events[events["user_id"].astype(str).str.startswith(prefix)]
    return tax, spec, app, events, mapping


def cmd_train(cfg, args):
    w = get_windows(cfg)
    tax, spec, app, events, mapping = _feature_inputs(cfg)
    ne = cfg["notification_events"]
    table = F.build_feature_table(app, events, mapping, spec, tax.categories, ne["sent"], ne["clicked"],
                                  w.feature_start, w.feature_end, w.lifetime_floor)
    labels = read_table(cfg["output"]["labels_file"])
    columns = F.feature_columns(spec, tax.categories)
    result = train_model(table, labels, columns, cfg)
    save_model(result, cfg["output"]["model_file"], cfg["output"]["model_config_file"])
    _write_json(result["metrics"], cfg["output"]["train_metrics_file"])
    result["importance"].to_csv(Path(cfg["paths"]["reports_dir"]) / "feature_importance.csv", index=False)
    m = result["metrics"]
    print(f"accuracy={m['accuracy']:.3f} majority_baseline={m['baselines']['majority_class_baseline']:.3f} "
          f"uniform_baseline={m['baselines']['uniform_baseline']:.3f}")


def _score(cfg, window_start: date, window_end: date):
    tax, spec, app, events, mapping = _feature_inputs(cfg)
    ne = cfg["notification_events"]
    w = get_windows(cfg)
    table = F.build_feature_table(app, events, mapping, spec, tax.categories, ne["sent"], ne["clicked"],
                                  window_start, window_end, w.lifetime_floor)
    booster, config = load_model(cfg["output"]["model_file"], cfg["output"]["model_config_file"])
    return events, mapping, predict_all(table, booster, config)


def cmd_predict(cfg, args):
    w = get_windows(cfg)
    run_date = datetime.strptime(args.run_date, "%Y-%m-%d").date() if args.run_date else date.today()
    end = run_date - timedelta(days=1)
    start = run_date - timedelta(days=w.rolling_days)
    freshness_gate(cfg["paths"]["notification_events_dir"], end.isoformat())
    _, _, preds = _score(cfg, start, end)

    out = Path(cfg["output"]["predictions_file"])
    previous = pd.read_parquet(out) if out.exists() else None
    moved = distribution_shift(preds, previous)
    if moved and not args.force:
        raise CCPError(f"Category share moved more than 10 points for {moved}. "
                       "Inspect the data, or rerun with --force to publish anyway.")
    write_atomic(preds, str(out))
    print(f"wrote {len(preds)} predictions to {out}")


def cmd_validate(cfg, args):
    w = get_windows(cfg)
    ne = cfg["notification_events"]
    events, mapping, preds = _score(cfg, w.feature_start, w.feature_end)
    result = forward_validation(events, mapping, preds, ne["clicked"], w.validation_start, w.validation_end)
    _write_json(result, cfg["output"]["validation_file"])
    print(f"hit_rate={result['hit_rate']:.3f} majority_baseline={result['majority_class_baseline']:.3f} "
          f"beats_majority={result['beats_majority_baseline']}")


def cmd_ingest(cfg, args):
    df = ingest_day(args.file, cfg, cfg.get("user_id_prefix", ""))
    out = write_day_idempotent(df, cfg["paths"]["notification_events_dir"], args.day)
    print(f"wrote {len(df)} rows to {out}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="ccp", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("command", choices=["classify", "labels", "train", "validate", "predict", "ingest"])
    p.add_argument("--config", default="config/pipeline.yaml")
    p.add_argument("--run-date", help="predict: YYYY-MM-DD (default today)")
    p.add_argument("--file", help="ingest: path to one raw JSON-lines file")
    p.add_argument("--day", help="ingest: YYYY-MM-DD the file belongs to")
    p.add_argument("--force", action="store_true", help="predict: publish even if category shares shifted")
    args = p.parse_args(argv)
    try:
        cfg = load_config(args.config)
        if args.command == "ingest" and not (args.file and args.day):
            raise CCPError("ingest needs --file and --day")
        {"classify": cmd_classify, "labels": cmd_labels, "train": cmd_train,
         "validate": cmd_validate, "predict": cmd_predict, "ingest": cmd_ingest}[args.command](cfg, args)
    except CCPError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
