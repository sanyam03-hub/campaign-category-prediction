# Data contract

All inputs are plain tables. Use any dataset you like (a public marketing or e-commerce
dataset, or a synthetic one you generate). Nothing in this repo assumes a particular source.
Files can be `.parquet` or `.csv`.

## `campaigns` (`paths.campaigns`)

| Column | Type | Notes |
|---|---|---|
| `campaign_name` | string | Unique key. Duplicates keep the first row. |
| `title` | string | Optional. Used only by the semantic fallback. |
| `message` | string | Optional. Used only by the semantic fallback. |

## `app_events` (`paths.app_events`)

| Column | Type | Notes |
|---|---|---|
| `user_id` | string | |
| `event` | string | Must match the names listed in `config/features.example.yaml`. |
| `event_date` | date or ISO string | |

## `notification events` (`paths.notification_events_dir`)

One file per day, named `events_YYYY-MM-DD.parquet` (or `.csv`). All files are concatenated.

| Column | Type | Notes |
|---|---|---|
| `user_id` | string | |
| `campaign_name` | string | Joined to the mapping by exact name. |
| `event_name` | string | Two values are used: the names in `notification_events.sent` and `.clicked`. |
| `event_date` | date or ISO string | |

## Windows (three consecutive, non-overlapping)

```
|---- feature window ----|---- label window ----|---- validation window ----|
 features are computed     the label is the most   predictions are checked
 here, and only here       clicked category here   against clicks here
```

`ccp` refuses to start if the feature window does not end before the label window starts.

## Minimum data for a meaningful run

- At least a few thousand users with clicks in the label window, and at least
  `model.min_class_samples` (default 10) per category. Training stops with a clear message
  if a category is too thin. Merge or drop thin categories in the taxonomy rather than lowering the limit.
- Click-through is naturally low in real notification data. If your dataset has clicks
  concentrated in one category, read `docs/lessons-learned.md` before trusting any accuracy number.
