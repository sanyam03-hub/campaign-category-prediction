#!/usr/bin/env bash
# Run before every push. Exits non-zero if anything that looks like company data is found.
# Edit the COMPANY_TERMS line to add names that must never appear in this repo.
set -u
cd "$(dirname "$0")/.."

COMPANY_TERMS='niyo|goniyo|niyo_dwh|kpi_dispatcher|sse_goa|clevertap_reports|de-databricks|niyo-de-prod'
fail=0

check() {  # $1 = label, $2 = regex
  hits=$(grep -rniE "$2" . \
    --exclude-dir=.git --exclude-dir=.pytest_cache --exclude-dir=__pycache__ \
    --exclude-dir=.venv --exclude=scan_repo.sh 2>/dev/null)
  if [ -n "$hits" ]; then
    echo "FOUND [$1]:"; echo "$hits" | head -20; echo; fail=1
  fi
}

check "company names / internal systems" "$COMPANY_TERMS"
check "S3 / bucket URLs"                 "s3a?://"
check "OA-style user IDs"                "\bOA[0-9A-F]{32}\b"
check "12-digit numeric IDs (91...)"     "\b91[0-9]{10}\b"
check "emails"                           "[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
check "tokens / keys"                    "(api[_-]?key|secret|passwd|password|bearer|AKIA[0-9A-Z]{16}|dapi[0-9a-f]{20})"

echo "--- tracked-looking data files ---"
find . \( -name '*.csv' -o -name '*.parquet' -o -name '*.pkl' -o -name '*.joblib' -o -name '*.json.gz' \) \
  -not -path './.git/*' -not -path './.venv/*' | while read -r f; do echo "DATA FILE: $f"; fail=1; done

if command -v gitleaks >/dev/null 2>&1; then
  gitleaks detect --no-git --source . || fail=1
else
  echo "(gitleaks not installed: 'brew install gitleaks' is worth doing before the first push)"
fi

[ "$fail" -eq 0 ] && echo "OK: nothing suspicious found." || { echo "STOP: fix the items above before pushing."; exit 1; }
