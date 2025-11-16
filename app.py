# app.py
import os
import json
import time
import io
import threading
from collections import Counter, defaultdict
from datetime import datetime

import numpy as np
import pandas as pd
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from flask import Flask, jsonify, render_template, request, redirect, session, url_for
from flask_socketio import SocketIO
from dotenv import load_dotenv

# GitHub storage helpers (create github_utils.py as provided earlier)
from github_utils import (
    github_load_json,
    github_load_csv,
    github_save_json,
)

load_dotenv()

# ---------------------------
# Config / Environment
# ---------------------------
# Repo layout (as you described)
# root: players.json, playlist.csv
# data_analysis/: analysis_results.json, artists.csv, distances.csv, embeddings.csv, neighbors.csv, person_artist_genres.json, tracks.csv

GITHUB_OWNER = os.getenv("GITHUB_REPO_OWNER")
GITHUB_REPO = os.getenv("GITHUB_REPO_NAME")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# App secrets & admin password for leaderboard reset
SECRET_KEY = os.getenv("SECRET_KEY", "dev_key")
RESET_PASSWORD = os.getenv("RESET_PASSWORD", "")

# Spotify
SPOTIPY_CLIENT_ID = os.getenv("SPOTIFY_CLIENT_ID")
SPOTIPY_CLIENT_SECRET = os.getenv("SPOTIFY_CLIENT_SECRET")
SPOTIPY_REDIRECT_URI = os.getenv("SPOTIPY_REDIRECT_URI", "http://localhost:5050/callback")

# Flask + SocketIO
app = Flask(__name__)
app.secret_key = SECRET_KEY
socketio = SocketIO(app, async_mode="threading", cors_allowed_origins="*")


# Spotify OAuth setup
scope = "user-read-currently-playing user-read-playback-state"
sp_oauth = SpotifyOAuth(
    client_id=SPOTIPY_CLIENT_ID,
    client_secret=SPOTIPY_CLIENT_SECRET,
    redirect_uri=SPOTIPY_REDIRECT_URI,
    scope=scope,
)

# ---------------------------
# GitHub-backed load wrappers (with local fallback)
# ---------------------------
def _local_candidates_for(path_in_repo: str):
    """
    Map repo path to likely local file paths to try as fallback.
    Example: 'data_analysis/embeddings.csv' -> ['./data/analysis/embeddings.csv', './data/embeddings.csv', 'data_analysis/embeddings.csv']
    """
    basename = os.path.basename(path_in_repo)
    candidates = []
    # prefer project data/analysis mapping
    if path_in_repo.startswith("data_analysis/"):
        candidates.append(os.path.join("data", "analysis", basename))
        candidates.append(os.path.join("data", basename))
        candidates.append(path_in_repo)
    else:
        candidates.append(os.path.join("data", path_in_repo))
        candidates.append(path_in_repo)
        candidates.append(os.path.join("data", basename))
    return candidates

def github_load_json_with_fallback(path_in_repo: str) -> dict:
    try:
        data = github_load_json(path_in_repo)
        if isinstance(data, dict) and data:
            print(f"✅ Loaded JSON from GitHub: {path_in_repo}")
            return data
    except Exception as e:
        print(f"⚠️ github_load_json error for {path_in_repo}: {e}")

    # local fallback
    for p in _local_candidates_for(path_in_repo):
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    print(f"🔁 Falling back to local JSON: {p}")
                    return json.load(f)
            except Exception as e:
                print(f"❌ Failed to parse local JSON {p}: {e}")
    print(f"❌ Could not load JSON {path_in_repo} from GitHub or local.")
    return {}

def github_load_csv_with_fallback(path_in_repo: str) -> pd.DataFrame:
    try:
        df = github_load_csv(path_in_repo)
        if isinstance(df, pd.DataFrame) and not df.empty:
            print(f"✅ Loaded CSV from GitHub: {path_in_repo} (rows={len(df)})")
            return df
        # If github_load_csv returned empty DataFrame, still continue to fallback
    except Exception as e:
        print(f"⚠️ github_load_csv error for {path_in_repo}: {e}")

    for p in _local_candidates_for(path_in_repo):
        if os.path.exists(p):
            try:
                print(f"🔁 Falling back to local CSV: {p}")
                return pd.read_csv(p)
            except Exception as e:
                print(f"❌ Failed to read local CSV {p}: {e}")
    print(f"❌ Could not load CSV {path_in_repo} from GitHub or local; returning empty DataFrame.")
    return pd.DataFrame()

# Keep the function names used throughout your code so minimal changes are needed
def drive_load_json(filename: str, parent_id: str = None) -> dict:
    # map filenames to repo paths
    mapping = {
        "analysis.json": "data_analysis/analysis_results.json",
        "analysis_results.json": "data_analysis/analysis_results.json",
        "person_artist_genres.json": "data_analysis/person_artist_genres.json",
        "players.json": "players.json",
        "guessers.json": "players.json",  # if you actually use guessers.json, map accordingly
        "leaderboard.json": "leaderboard.json",
    }
    repo_path = mapping.get(filename, filename if not filename.startswith("data_analysis/") else filename)
    return github_load_json_with_fallback(repo_path)

def drive_load_csv(filename: str, parent_id: str = None) -> pd.DataFrame:
    # map analysis CSV names to data_analysis/
    analysis_files = {"embeddings.csv", "neighbors.csv", "nearest_neighbors.csv", "distances.csv", "artists.csv", "tracks.csv"}
    if filename in analysis_files:
        repo_path = f"data_analysis/{filename}" if not filename.startswith("data_analysis/") else filename
    else:
        repo_path = filename
    return github_load_csv_with_fallback(repo_path)

# ---------------------------
# Leaderboard persistence (save primary)
# ---------------------------
LEADERBOARD_REPO_PATH = "leaderboard.json"

def save_leaderboard_to_github(lb_dict: dict) -> bool:
    """Save leaderboard dict to repo root."""
    timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    payload = {"leaderboard": lb_dict, "updated_at_utc": timestamp}
    ok_primary = False
    try:
        ok_primary = github_save_json(LEADERBOARD_REPO_PATH, payload, message=f"Update leaderboard {timestamp}")
    except Exception as e:
        print(f"⚠️ Error saving leaderboard to GitHub: {e}")
    if ok_primary:
        print("✅ Leaderboard saved to GitHub.")
    else:
        print("⚠️ Failed to save primary leaderboard to GitHub.")
    return bool(ok_primary)

def load_leaderboard_from_github() -> dict:
    data = github_load_json_with_fallback(LEADERBOARD_REPO_PATH)
    if isinstance(data, dict):
        return data.get("leaderboard", {}) if data else {}
    return {}

def reset_leaderboard_on_github() -> dict:
    empty = {}
    saved = save_leaderboard_to_github(empty)
    if saved:
        print("✅ Leaderboard reset persisted to GitHub.")
    else:
        print("⚠️ Leaderboard reset not persisted to GitHub.")
    return empty

# ---------------------------
# Game Data (GitHub / local)
# ---------------------------
def _normalize_name(name: str) -> str:
    """Convert 'first_last' -> 'First Last'."""
    if not isinstance(name, str):
        return ""
    parts = name.strip().split("_")
    return " ".join(p.capitalize() for p in parts if p)


def load_guessers():
    data = drive_load_json("players.json", None)

    # players.json may contain "guessers" OR "players"
    raw = data.get("guessers") or []

    # Normalize names
    guessers = [_normalize_name(x) for x in raw]
    return sorted(guessers)


def load_data_owners():
    data = drive_load_json("players.json", None)

    # players.json may contain "guessers" OR "players"
    raw = data.get("data_owners")  or []

    # Normalize names
    guessers = [_normalize_name(x) for x in raw]
    return sorted(guessers)


def load_playlist():
    # playlist.csv is in the root folder
    df = drive_load_csv("playlist.csv")

    if df.empty:
        print("⚠️ playlist.csv not found or empty in GitHub or Drive.")
        return {}

    songs = {}
    for _, row in df.iterrows():
        song_id = row.get("ID") or row.get("id")
        owners_raw = row.get("people", "")
        owners = str(owners_raw).split(";")

        songs[song_id] = {
            "id": song_id,
            "name": row.get("Track", "") or row.get("track", ""),
            "artist": row.get("Artist", "") or row.get("artist", ""),
            "owners": owners,
        }

    print(f"✅ Loaded {len(songs)} songs from playlist.csv")
    return songs


# Load initial game data
SONG_DATA = load_playlist()
GUESSERS = load_guessers()
POSSIBLE_OWNERS = load_data_owners()


# ---------------------------
# Game State
# ---------------------------
current_song = {}
# initialize leaderboard from GitHub (fallback empty)
_initial_lb = load_leaderboard_from_github()
if _initial_lb:
    leaderboard = defaultdict(int, _initial_lb)
    print(f"✅ Loaded persisted leaderboard ({len(_initial_lb)} entries).")
else:
    leaderboard = defaultdict(int)
    print("ℹ️ Starting with empty leaderboard.")
all_guesses = defaultdict(list)
token_info = None
spotify_token = None
poller_started = False

# ---------------------------
# Spotify Auth + Polling
# ---------------------------
@app.route("/login")
def login():
    return redirect(sp_oauth.get_authorize_url())

@app.route("/callback")
def callback():
    global token_info, spotify_token, poller_started
    code = request.args.get("code")
    if not code:
        return "Missing code", 400
    # get_access_token is deprecated in some spotipy versions; handle both
    try:
        token_info_local = sp_oauth.get_access_token(code)
    except TypeError:
        token_info_local = sp_oauth.get_access_token(code)
    token_info = token_info_local
    spotify_token = token_info.get("access_token")
    try:
        session["token_info"] = token_info
    except Exception:
        pass
    print("✅ Spotify token acquired.")
    if not poller_started:
        threading.Thread(target=poll_spotify, daemon=True).start()
        poller_started = True
        print("🔔 Started Spotify poller thread.")
    return redirect(url_for("game"))

def get_spotify_client():
    global token_info, spotify_token
    if not token_info:
        return None
    try:
        # some spotipy versions use is_token_expired(token_info)
        if sp_oauth.is_token_expired(token_info):
            print("🔁 Refreshing token...")
            token_info = sp_oauth.refresh_access_token(token_info["refresh_token"])
            spotify_token = token_info.get("access_token")
            print("✅ Token refreshed.")
    except Exception as e:
        print("⚠️ Token refresh error:", e)
        return None
    return spotipy.Spotify(auth=spotify_token)

def poll_spotify():
    global current_song
    print("🔔 Spotify poller running.")
    first_emit_done = False
    previous_song = None
    while True:
        try:
            sp = get_spotify_client()
            if not sp:
                time.sleep(5)
                continue
            try:
                playback = sp.current_playback()
            except Exception as e:
                print("⚠️ Error calling current_playback():", e)
                playback = None

            if playback and playback.get("item"):
                track = playback["item"]
                song_id = track.get("id")
                name = track.get("name")
                artist = ", ".join(a["name"] for a in track.get("artists", []))
                album_images = track.get("album", {}).get("images", [])
                album_image = album_images[0]["url"] if album_images else None

                new_song = {"id": song_id, "name": name, "artist": artist, "image": album_image}

                if not first_emit_done or current_song.get("id") != song_id:
                    print(f"🎶 Song update → {name} — {artist}")
                    previous_song = current_song.copy() if current_song else None
                    current_song = new_song
                    previous_reveal = None
                    if previous_song and previous_song.get("id"):
                        prev_id = previous_song["id"]
                        owners = SONG_DATA.get(prev_id, {}).get("owners", [])
                        guesses = [g for _, g in all_guesses.get(prev_id, [])]
                        most_common, votes = (None, 0)
                        if guesses:
                            counter = Counter(guesses)
                            most_common, votes = counter.most_common(1)[0]
                        previous_reveal = {"previous_song": previous_song, "owners": owners, "most_common": most_common, "votes": votes}
                        all_guesses.pop(prev_id, None)

                    payload = {"current_song": current_song}
                    if previous_reveal:
                        payload["previous_reveal"] = previous_reveal

                    socketio.emit("song_update", payload)
                    first_emit_done = True
            else:
                # nothing playing
                # emit a heartbeat occasionally?
                pass
        except Exception as e:
            print("⚠️ Error polling Spotify (outer):", e)
        time.sleep(5)

@socketio.on("connect")
def handle_connect():
    if current_song.get("id"):
        socketio.emit("song_update", {"current_song": current_song})

threading.Thread(target=poll_spotify, daemon=True).start()
poller_started = True

# ---------------------------
# Routes
# ---------------------------
@app.route("/")
def game():
    return render_template("game.html", owners=POSSIBLE_OWNERS, guessers=GUESSERS)

@app.route("/display")
def display():
    return render_template("display.html")

@app.route("/leaderboard")
def show_leaderboard():
    sorted_board = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
    return render_template("leaderboard.html", leaderboard=sorted_board)

@app.route("/api/guess", methods=["POST"])
def handle_guess():
    data = request.get_json(force=True)
    song_id = data.get("song_id")
    username = data.get("username")
    guess = data.get("guess")
    if not song_id or not username or not guess:
        return jsonify({"success": False, "message": "Missing fields"})
    user_guesses = [u for u, _ in all_guesses.get(song_id, [])]
    if username in user_guesses:
        return jsonify({"success": False, "message": "You’ve already guessed for this song!"})
    all_guesses[song_id].append((username, guess))
    correct = False
    if guess in SONG_DATA.get(song_id, {}).get("owners", []):
        leaderboard[username] += 1
        correct = True
        # Attempt to persist leaderboard (best-effort)
        try:
            save_ok = save_leaderboard_to_github(dict(leaderboard))
            if not save_ok:
                print("⚠️ Leaderboard save returned False")
        except Exception as e:
            print(f"⚠️ Failed to persist leaderboard: {e}")
    return jsonify({"success": True, "correct": correct, "message": "Guess submitted!"})

# ---------------------------
# Analysis Route
# ---------------------------
@app.route("/analysis")
def analysis():
    # Load analysis files (prefer data_analysis/*)
    data = drive_load_json("analysis_results.json", None) or drive_load_json("analysis.json", None)
    embeddings = drive_load_csv("embeddings.csv", None)
    neighbors = drive_load_csv("neighbors.csv", None)
    neighbors['user'] = neighbors['Unnamed: 0']
    neighbors = neighbors.drop(columns=['Unnamed: 0'])

    # fallback name used earlier in your app: nearest_neighbors.csv
    if neighbors.empty:
        neighbors = drive_load_csv("nearest_neighbors.csv", None)
    distances = drive_load_csv("distances.csv", None)
    person_genres = drive_load_json("person_artist_genres.json", None)

    print("✅ All analysis load attempts finished.")

    # Prepare main stats
    avg_release_date_raw = data.get("average_release_date", "") or ""
    avg_release_date = avg_release_date_raw.split(" ")[0] if avg_release_date_raw else ""
    analysis_data = {
        "average_song_length_minutes": round(data.get("average_song_length", 0), 2),
        "average_popularity": data.get("average_popularity", 0),
        "average_release_date": avg_release_date,
        "most_common_track": data.get("most_common_track", "Unknown Track"),
        "most_common_track_artist": data.get("most_common_track_artist", "Unknown Artist"),
        "most_common_track_count": data.get("most_common_track_count", 0),
        "most_common_artist": data.get("most_common_artist", "Unknown Artist"),
        "most_common_artist_count": data.get("most_common_artist_count", 0),
        "most_common_genre": data.get("most_common_genre", "Unknown"),
        "most_common_genre_count": data.get("most_common_genre_count", 0),
    }

    # Normalize neighbors globally (closest+furthest combined min/max)
    if not neighbors.empty:
        try:
            # STRICT column matching (avoid closest_point)
            closest_col = "closest_distance" if "closest_distance" in neighbors.columns else None
            furthest_col = "furthest_distance" if "furthest_distance" in neighbors.columns else None

            if closest_col and furthest_col:
                # float conversion safe now
                all_vals = np.concatenate([
                    neighbors[closest_col].astype(float).values,
                    neighbors[furthest_col].astype(float).values,
                ])

                dmin = np.nanmin(all_vals)
                dmax = np.nanmax(all_vals)
                span = (dmax - dmin) if (dmax - dmin) != 0 else 1.0

                neighbors["closest_sim"] = 1 - ((neighbors[closest_col] - dmin) / span)
                neighbors["furthest_sim"] = 1 - ((neighbors[furthest_col] - dmin) / span)

                neighbors["closest_sim"] = neighbors["closest_sim"].clip(0, 1)
                neighbors["furthest_sim"] = neighbors["furthest_sim"].clip(0, 1)

                print("✅ Normalized neighbor distances (global).")
            else:
                print("⚠️ Required distance columns not found.")
                neighbors["closest_sim"] = np.nan
                neighbors["furthest_sim"] = np.nan

        except Exception as e:
            print(f"⚠️ Error normalizing neighbors: {e}")
            neighbors["closest_sim"] = np.nan
            neighbors["furthest_sim"] = np.nan


    # Convert DataFrames to dicts for template
    embeddings_out = embeddings.to_dict(orient="records") if not embeddings.empty else []
    neighbors_out = neighbors.to_dict(orient="records") if not neighbors.empty else []
    distances_out = {}
    if not distances.empty:
        try:
            # If distances is square with index labels, try to set index if necessary
            if distances.columns[0].lower() in ("unnamed: 0", "index"):
                distances = distances.set_index(distances.columns[0])
            distances_out = distances.fillna("").to_dict()
        except Exception as e:
            print(f"⚠️ Could not convert distances.csv cleanly: {e}")
            distances_out = distances.to_dict()
    else:
        distances_out = {}

    return render_template(
        "analysis.html",
        data=analysis_data,
        embeddings=embeddings_out,
        neighbors=neighbors_out,
        distances=distances_out,
        person_genres=person_genres,
    )

# ---------------------------
# Leaderboard Reset (password protected)
# ---------------------------
@app.route("/reset_leaderboard", methods=["POST"])
def handle_reset_leaderboard():
    # Accept JSON or form data
    payload = {}
    try:
        payload = request.get_json(force=False) or {}
    except Exception:
        payload = request.form.to_dict() or {}
    pw = payload.get("password") or request.form.get("password") or ""
    if not RESET_PASSWORD:
        return jsonify({"success": False, "message": "Reset password not configured on server."}), 403
    if pw != RESET_PASSWORD:
        return jsonify({"success": False, "message": "Wrong password"}), 403

    # clear in-memory leaderboard and persist
    global leaderboard
    leaderboard = defaultdict(int)
    saved = save_leaderboard_to_github(dict(leaderboard))
    if not saved:
        return jsonify({"success": False, "message": "Leaderboard reset but failed to persist to GitHub."}), 500
    return jsonify({"success": True, "message": "Leaderboard reset"}), 200

if __name__ == "__main__":
    socketio.run(app, debug=True, host="127.0.0.1", port=5050)