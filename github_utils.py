# github_utils.py
import os
import json
import base64
import requests
import io
import pandas as pd
from typing import Optional

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GITHUB_OWNER = os.getenv("GITHUB_REPO_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO_NAME")

API_BASE = "https://api.github.com/repos"

def _headers():
    h = {"Accept": "application/vnd.github.v3+json"}
    if GITHUB_TOKEN:
        h["Authorization"] = f"token {GITHUB_TOKEN}"
    return h

def _contents_url(path: str) -> str:
    # path should be repository path (no leading slash)
    path = path.lstrip("/")
    return f"{API_BASE}/{GITHUB_OWNER}/{GITHUB_REPO}/contents/{path}"

def github_get_file(path: str) -> Optional[dict]:
    """
    Returns the JSON response from the GitHub Contents API for the file/folder,
    or None if not found or on error.
    """
    if not (GITHUB_OWNER and GITHUB_REPO):
        print("⚠️ GITHUB_REPO_OWNER / GITHUB_REPO_NAME not set.")
        return None

    url = _contents_url(path)
    r = requests.get(url, headers=_headers())
    if r.status_code == 200:
        return r.json()
    else:
        # If folder, GitHub returns 200 with list; 404 if not found
        print(f"⚠️ GitHub GET {path} returned {r.status_code}")
        return None

def github_load_json(path: str) -> dict:
    """
    Load JSON file from GitHub repo path (e.g. 'players.json' or 'data_analysis/analysis_results.json').
    Falls back to {} on any failure.
    """
    resp = github_get_file(path)
    if not resp:
        return {}

    # If it's a file object
    if isinstance(resp, dict) and resp.get("content"):
        content = base64.b64decode(resp["content"]).decode("utf-8")
        try:
            return json.loads(content)
        except Exception as e:
            print(f"⚠️ Failed to parse JSON from GitHub {path}: {e}")
            return {}
    # Unexpected type
    print(f"⚠️ Unexpected GitHub response for JSON path {path}: {type(resp)}")
    return {}

def github_load_csv(path: str) -> pd.DataFrame:
    """
    Load CSV from GitHub and return pandas DataFrame.
    On failure, returns empty DataFrame.
    """
    resp = github_get_file(path)
    if not resp:
        return pd.DataFrame()

    if isinstance(resp, dict) and resp.get("content"):
        content = base64.b64decode(resp["content"]).decode("utf-8")
        try:
            return pd.read_csv(io.StringIO(content))
        except Exception as e:
            print(f"⚠️ Failed to parse CSV from GitHub {path}: {e}")
            return pd.DataFrame()
    print(f"⚠️ Unexpected GitHub response for CSV path {path}: {type(resp)}")
    return pd.DataFrame()

def github_save_file(path: str, content_bytes: bytes, message: str = "Update via webapp"):
    """
    Save a file to GitHub repo (create or update). content_bytes are the raw bytes.
    Returns True on success, False on failure.
    """
    if not (GITHUB_OWNER and GITHUB_REPO and GITHUB_TOKEN):
        print("⚠️ Missing GITHUB_OWNER/REPO/TOKEN for writing.")
        return False

    url = _contents_url(path)
    # Check if file exists to fetch sha
    r = requests.get(url, headers=_headers())
    sha = None
    if r.status_code == 200:
        existing = r.json()
        sha = existing.get("sha")
    elif r.status_code not in (404,):
        print(f"⚠️ GitHub HEAD check for {path} returned {r.status_code}")
        return False

    b64 = base64.b64encode(content_bytes).decode("utf-8")
    payload = {
        "message": message,
        "content": b64,
    }
    if sha:
        payload["sha"] = sha

    put = requests.put(url, headers=_headers(), json=payload)
    if put.status_code in (200, 201):
        print(f"✅ Saved to GitHub: {path}")
        return True
    else:
        print(f"⚠️ GitHub save failed ({put.status_code}): {put.text}")
        return False

def github_save_json(path: str, data: dict, message: str = "Update JSON via webapp"):
    b = json.dumps(data, indent=2).encode("utf-8")
    return github_save_file(path, b, message)

def github_save_csv(path: str, df: pd.DataFrame, message: str = "Update CSV via webapp"):
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    return github_save_file(path, csv_bytes, message)
