#!/bin/sh
# Wait for any running training job to finish, then (re)start the sweep; finished runs are skipped.
cd "$(dirname "$0")/.." || exit 1
while pgrep -f "[k]eepspan.train --model" > /dev/null; do sleep 10; done
exec .venv/bin/python scripts/sweep.py --need-gb 3.6
