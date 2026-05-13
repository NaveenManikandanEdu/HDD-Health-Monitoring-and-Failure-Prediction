# ml/preprocessing/add_labels.py
"""
Deterministic label generation for HDD datasets.

Generates:
 - failure_next_day
 - failure_next_7_days

Input:
  timelines = {
      device_id: [
          {"date": Timestamp, "failure": 0/1},
          ...
      ]
  }
"""

from __future__ import annotations
import pandas as pd
from collections import defaultdict
from typing import Dict, List


# ---------------------------------------------------------
# Build label map from timelines
# ---------------------------------------------------------
def build_label_map_from_timelines(
    timelines: Dict[str, List[Dict]]
) -> Dict[str, Dict[str, Dict[str, int]]]:
    """
    Convert per-device timelines into:

    {
      "YYYY-MM-DD": {
          "device_id": {
              "failure_next_day": 0/1,
              "failure_next_7_days": 0/1
          }
      }
    }
    """

    date_label_map = defaultdict(dict)

    for device_id, events in timelines.items():
        if not events:
            continue

        # keep valid dates only
        events = [e for e in events if pd.notna(e["date"])]
        if not events:
            continue

        # sort chronologically
        events = sorted(events, key=lambda x: x["date"])

        failures = [int(e["failure"]) for e in events]
        dates = [e["date"].strftime("%Y-%m-%d") for e in events]

        n = len(events)

        for i in range(n):
            next_day = 1 if (i + 1 < n and failures[i + 1] == 1) else 0
            next_7 = 1 if any(failures[i + 1 : i + 8]) else 0

            date_label_map[dates[i]][device_id] = {
                "failure_next_day": next_day,
                "failure_next_7_days": next_7,
            }

    return date_label_map
