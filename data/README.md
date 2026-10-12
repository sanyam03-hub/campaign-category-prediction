# data/

This folder is intentionally empty. Put your own dataset here (it is git-ignored).

Expected layout (see `docs/data-contract.md` for the column definitions):

```
data/
├── campaigns.csv            one row per campaign: campaign_name, title, message
├── app_events.parquet       user_id, event, event_date
├── events/                  notification events, one file per day
│   ├── events_2024-01-01.parquet
│   └── ...
└── raw/events/              optional: raw JSON-lines exports for `ingest`
```

Never commit real customer data. Before pushing, run `scripts/scan_repo.sh`.
