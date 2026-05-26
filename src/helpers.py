"""
src/helpers.py

Shared logic for the image2playlist pipeline.
Used by both demo/app.py and notebooks/03_vibe_mapping.ipynb.

Exports
-------
VIBE_TAXONOMY       dict   — 12 vibe definitions (label, probe, emoji)
diverse_top_k()     fn     — artist-capped, genre-spread song selection
build_playlist()    fn     — encode image → detect vibe → rank songs
spotify_search_track()        fn — look up a single track on Spotify
enrich_with_spotify()         fn — add preview/URL columns to a playlist DataFrame
"""

import io
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

# ── vibe taxonomy ─────────────────────────────────────────────────────────────

VIBE_TAXONOMY: dict[str, dict] = {
    "cozy_cafe": {
        "label": "Cozy Cafe",
        "probe": "a cozy coffee shop with warm lighting and soft background music",
        "emoji": "☕️",
    },
    "winter_snow": {
        "label": "Winter / Snow",
        "probe": "a quiet snowy street on a cold winter evening",
        "emoji": "❄️",
    },
    "beach_summer": {
        "label": "Beach / Summer",
        "probe": "a bright sunny beach in summer with people relaxing",
        "emoji": "🏖️",
    },
    "rainy_melancholic": {
        "label": "Rainy / Melancholic",
        "probe": "a rainy night alone in the city feeling nostalgic and sad",
        "emoji": "🌧️",
    },
    "party_energetic": {
        "label": "Party / Energetic",
        "probe": "a high-energy dance party with flashing lights and a dancing crowd",
        "emoji": "🎉",
    },
    "romantic_evening": {
        "label": "Romantic Evening",
        "probe": "a candlelit romantic dinner setting at night",
        "emoji": "🕯️",
    },
    "nature_peaceful": {
        "label": "Nature / Peaceful",
        "probe": "a peaceful walk through green forests and nature trails",
        "emoji": "🌿",
    },
    "urban_hustle": {
        "label": "Urban / Hustle",
        "probe": "a busy city street at midday with crowds and tall buildings",
        "emoji": "🏙️",
    },
    "late_night_drive": {
        "label": "Late Night Drive",
        "probe": "driving alone on a dark highway at night with city lights in the distance",
        "emoji": "🌃",
    },
    "morning_calm": {
        "label": "Morning / Calm",
        "probe": "a calm peaceful morning at home with soft sunlight coming through the window",
        "emoji": "🌅",
    },
    "dark_moody": {
        "label": "Dark / Moody",
        "probe": "a dark moody aesthetic with deep shadows and muted tones",
        "emoji": "🖤",
    },
    "festival_concert": {
        "label": "Festival / Concert",
        "probe": "an outdoor music festival with a large crowd and a stage",
        "emoji": "🎸",
    },
}

VIBE_KEYS   = list(VIBE_TAXONOMY.keys())
VIBE_LABELS = [VIBE_TAXONOMY[k]["label"] for k in VIBE_KEYS]
VIBE_PROBES = [VIBE_TAXONOMY[k]["probe"] for k in VIBE_KEYS]
VIBE_EMOJIS = [VIBE_TAXONOMY[k]["emoji"] for k in VIBE_KEYS]

KIDS_GENRES = "children|kids|disney"


# ── pipeline ──────────────────────────────────────────────────────────────────

def diverse_top_k(
    sims: np.ndarray,
    index_df: pd.DataFrame,
    top_k: int = 10,
    max_per_artist: int = 2,
    min_genre_spread: int = 3,
) -> list[int]:
    """
    Return up to top_k song indices satisfying diversity constraints:
      - max_per_artist  : no primary artist contributes more than this many tracks
      - min_genre_spread: extend selection until this many distinct genre tokens
                          are covered (soft cap at 1.5 * top_k)
    """
    ranked        = np.argsort(sims)[::-1]
    selected      = []
    artist_counts: dict[str, int] = {}
    covered_genres: set[str] = set()

    for idx in ranked:
        if len(selected) >= top_k:
            if len(covered_genres) >= min_genre_spread:
                break
            if len(selected) >= int(top_k * 1.5):
                break

        row = index_df.iloc[int(idx)]
        primary = (
            str(row["artists"])
            .split(",")[0].split(";")[0]
            .split("(feat")[0].split("feat.")[0]
            .strip().lower()
        )
        if artist_counts.get(primary, 0) >= max_per_artist:
            continue

        selected.append(int(idx))
        artist_counts[primary] = artist_counts.get(primary, 0) + 1
        covered_genres |= {
            g.strip().lower()
            for g in str(row.get("merged_genres", "")).split(",")
            if g.strip()
        }

    return selected


def build_playlist(
    image_source,
    model,
    preprocess,
    device: str,
    song_emb: np.ndarray,
    index_df: pd.DataFrame,
    vibe_emb: np.ndarray,
    sp=None,
    top_k: int = 10,
    max_per_artist: int = 2,
    min_genre_spread: int = 3,
    enrich_spotify: bool = True,
) -> tuple:
    """
    Full image-to-playlist pipeline.

    Parameters
    ----------
    image_source : str | bytes | file-like
        File path, raw bytes, or any object PIL.Image.open() accepts.
    model, preprocess, device
        CLIP model and preprocessing function.
    song_emb : np.ndarray  shape [N, 512]
        Pre-normalised song text embeddings.
    index_df : pd.DataFrame
        Song metadata index aligned with song_emb.
    vibe_emb : np.ndarray  shape [12, 512]
        Pre-normalised vibe probe embeddings.
    sp : spotipy.Spotify | None
        Client-credentials Spotify client; if None, Spotify enrichment is skipped.
    enrich_spotify : bool
        Whether to call Spotify for preview URLs / links.

    Returns
    -------
    (vibe_key, vibe_label, vibe_emoji, vibe_score, all_vibe_sims, playlist_df)
    """
    # Step 1 — encode image
    if isinstance(image_source, (bytes, bytearray)):
        pil_img = Image.open(io.BytesIO(image_source)).convert("RGB")
    elif isinstance(image_source, str):
        try:
            pil_img = Image.open(image_source).convert("RGB")
        except Exception as e:
            raise ValueError(
                f"Could not open '{image_source}'.\n"
                f"iPhone HEIC photos: re-save as JPEG via Share → Save to Files.\n"
                f"Original error: {e}"
            ) from e
    else:
        pil_img = Image.open(image_source).convert("RGB")

    img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        img_enc = model.encode_image(img_tensor)
        img_enc = F.normalize(img_enc, dim=-1).cpu().numpy()   # [1, 512]

    # Step 2 — detect vibe
    vibe_sims    = (vibe_emb @ img_enc.T).squeeze()            # [12]
    top_idx      = int(vibe_sims.argmax())
    top_key      = VIBE_KEYS[top_idx]
    top_label    = VIBE_LABELS[top_idx]
    top_emoji    = VIBE_EMOJIS[top_idx]
    top_score    = float(vibe_sims[top_idx])

    # Step 3 — cosine similarity vs full song pool
    song_sims    = (song_emb @ img_enc.T).squeeze()            # [N]

    # Step 4 — diverse selection
    selected_idx = diverse_top_k(song_sims, index_df, top_k, max_per_artist, min_genre_spread)

    rows = []
    for rank, idx in enumerate(selected_idx):
        r = index_df.iloc[idx]
        rows.append({
            "rank"       : rank + 1,
            "similarity" : round(float(song_sims[idx]), 4),
            "track_name" : r["track_name"],
            "artists"    : r["artists"],
            "genres"     : r["merged_genres"],
            "popularity" : int(r["popularity"]),
        })
    playlist = pd.DataFrame(rows)

    # Step 5 — Spotify enrichment
    if enrich_spotify and sp is not None and not playlist.empty:
        playlist = enrich_with_spotify(sp, playlist)

    return top_key, top_label, top_emoji, top_score, vibe_sims, playlist


# ── Spotify helpers ───────────────────────────────────────────────────────────

_spotify_cache: dict = {}


def spotify_search_track(sp, track_name: str, artist: str, retries: int = 2) -> dict | None:
    """
    Search Spotify for a single track.

    Returns a dict with keys:
        id, name, artists, album, album_art, uri, external_url, preview_url
    Returns None if not found. Results are cached within the process lifetime.
    """
    if sp is None:
        return None

    key = (track_name.lower().strip(), artist.lower().strip())
    if key in _spotify_cache:
        return _spotify_cache[key]

    clean_name     = track_name.split("(feat")[0].split("(with")[0].strip()
    primary_artist = artist.split(",")[0].split(";")[0].strip()
    query          = f'track:"{clean_name}" artist:"{primary_artist}"'

    for _ in range(retries + 1):
        try:
            results = sp.search(q=query, type="track", limit=1)
            items   = results.get("tracks", {}).get("items", [])
            if items:
                t = items[0]
                hit = {
                    "id"          : t["id"],
                    "name"        : t["name"],
                    "artists"     : ", ".join(a["name"] for a in t["artists"]),
                    "album"       : t["album"]["name"],
                    "album_art"   : (t["album"]["images"][0]["url"]
                                     if t["album"]["images"] else None),
                    "uri"         : t["uri"],
                    "external_url": t["external_urls"].get("spotify", ""),
                    "preview_url" : t.get("preview_url"),
                }
                _spotify_cache[key] = hit
                return hit
        except Exception as e:
            if getattr(e, "http_status", None) == 429:
                retry_after = int(getattr(e, "headers", {}).get("Retry-After", 2))
                time.sleep(retry_after + 1)
            else:
                break

    _spotify_cache[key] = None
    return None


def enrich_with_spotify(sp, playlist_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add spotify_id, spotify_url, spotify_uri, preview_url, album_art columns
    to a playlist DataFrame. Tracks not found on Spotify receive None.
    """
    df = playlist_df.copy()
    ids, urls, uris, previews, arts = [], [], [], [], []

    for _, row in df.iterrows():
        hit = spotify_search_track(sp, row["track_name"], row["artists"])
        ids.append(hit["id"]           if hit else None)
        urls.append(hit["external_url"] if hit else None)
        uris.append(hit["uri"]          if hit else None)
        previews.append(hit["preview_url"] if hit else None)
        arts.append(hit["album_art"]    if hit else None)

    df["spotify_id"]  = ids
    df["spotify_url"] = urls
    df["spotify_uri"] = uris
    df["preview_url"] = previews
    df["album_art"]   = arts
    return df
