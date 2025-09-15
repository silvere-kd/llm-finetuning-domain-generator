# src/utils_json.py

import json, re
from typing import List
from templates.constants import JSON_ARRAY_REGEX

def extract_json_array(text: str) -> List[str]:
    """Utilities for robust JSON array extraction from model text outputs.
    Return last JSON array from text as list[str]; [] on failure."""
    matches = re.findall(JSON_ARRAY_REGEX, text, flags=re.S)
    if not matches:
        return []
    for candidate in reversed(matches):
        norm = candidate.strip()
        # Soft fix for single quotes
        if "'" in norm and '"' not in norm:
            norm = norm.replace("'", '"')
        try:
            parsed = json.loads(norm)
            if isinstance(parsed, list):
                out = []
                for x in parsed:
                    if isinstance(x, (str, int, float)):
                        out.append(str(x).strip().lower())
                # dedup preserve order
                seen, dedup = set(), []
                for d in out:
                    if d not in seen:
                        seen.add(d)
                        dedup.append(d)
                return dedup
        except json.JSONDecodeError:
            continue
    return []
