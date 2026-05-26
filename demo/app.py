"""
demo/app.py — Image-to-Playlist Streamlit App

Run from the repo root:
    streamlit run demo/app.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import streamlit as st

# Make src/ importable when running from any working directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Load .env before anything reads os.getenv — must be at module level so
# Streamlit picks it up when it imports this file (not just __main__).
try:
    from dotenv import load_dotenv
    _env = ROOT / ".env"
    if _env.exists():
        load_dotenv(_env)
except ImportError:
    pass

from src.helpers import (
    VIBE_TAXONOMY,
    VIBE_KEYS,
    VIBE_LABELS,
    VIBE_EMOJIS,
    KIDS_GENRES,
    build_playlist,
)

# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="image2playlist",
    page_icon="🎵",
    layout="wide",
)

DATA_DIR = ROOT / "data"

SONG_EMBEDDINGS_PATH = DATA_DIR / "song_embeddings.npy"
SONG_INDEX_PATH      = DATA_DIR / "song_index.csv"


# ── cached resource loading ───────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading CLIP model and song data…")
def load_resources():
    import clip
    import pandas as pd
    import spotipy
    from spotipy.oauth2 import SpotifyClientCredentials

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    # Song embeddings + metadata (filter kids genres at load time)
    song_emb_all = np.load(SONG_EMBEDDINGS_PATH)
    index_df_all = pd.read_csv(SONG_INDEX_PATH)

    keep     = ~index_df_all["merged_genres"].str.contains(KIDS_GENRES, case=False, na=False)
    song_emb = song_emb_all[keep.values]
    index_df = index_df_all[keep.values].reset_index(drop=True)

    # Precompute vibe probe embeddings [12, 512]
    with torch.no_grad():
        tokens   = clip.tokenize([VIBE_TAXONOMY[k]["probe"] for k in VIBE_KEYS], truncate=True).to(device)
        vibe_emb = model.encode_text(tokens)
        import torch.nn.functional as F
        vibe_emb = F.normalize(vibe_emb, dim=-1).cpu().numpy()

    # Spotify client (client credentials — no user login needed)
    client_id     = os.getenv("SPOTIPY_CLIENT_ID", "")
    client_secret = os.getenv("SPOTIPY_CLIENT_SECRET", "")
    sp = None
    if client_id and client_secret:
        try:
            sp = spotipy.Spotify(
                auth_manager=SpotifyClientCredentials(
                    client_id=client_id,
                    client_secret=client_secret,
                )
            )
        except Exception:
            sp = None

    return device, model, preprocess, song_emb, index_df, vibe_emb, sp


# ── UI ────────────────────────────────────────────────────────────────────────
def main():
    st.title("🎵 image2playlist")
    st.markdown(
        "Upload a photo and we'll detect its **vibe** using CLIP, "
        "then generate a matching playlist from 60K+ Spotify tracks."
    )
    st.divider()

    resources = load_resources()
    device, model, preprocess, song_emb, index_df, vibe_emb, sp = resources

    if sp is None:
        st.warning(
            "Spotify credentials not set — previews and links won't be available. "
            "Add `SPOTIPY_CLIENT_ID` and `SPOTIPY_CLIENT_SECRET` to your `.env` file and restart.",
            icon="⚠️",
        )

    uploaded = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        help="JPG / PNG / WebP supported. iPhone HEIC: re-save as JPEG first.",
    )

    if not uploaded:
        st.info("Upload an image above to get started.", icon="👆")
        return

    image_bytes = uploaded.read()

    col_img, col_vibe = st.columns([1, 1], gap="large")

    with col_img:
        st.image(image_bytes, use_container_width=True, caption=uploaded.name)

    with col_vibe:
        with st.spinner("Analyzing image and building playlist…"):
            try:
                top_key, top_label, top_emoji, top_score, vibe_sims, playlist = build_playlist(
                    image_source=image_bytes,
                    model=model,
                    preprocess=preprocess,
                    device=device,
                    song_emb=song_emb,
                    index_df=index_df,
                    vibe_emb=vibe_emb,
                    sp=sp,
                )
            except Exception as e:
                st.error(f"Pipeline error: {e}")
                return

        st.subheader("Detected vibe")
        st.markdown(f"<h2 style='margin:0'>{top_emoji} {top_label}</h2>", unsafe_allow_html=True)
        st.caption(f"Cosine similarity: {top_score:.3f}")
        st.markdown("")

        # Vibe bar chart (sorted descending)
        sorted_idx = np.argsort(vibe_sims)[::-1]
        chart_df = pd.DataFrame({
            "Vibe" : [f"{VIBE_EMOJIS[i]} {VIBE_LABELS[i]}" for i in sorted_idx],
            "Score": [round(float(vibe_sims[i]), 4)         for i in sorted_idx],
        })
        st.bar_chart(chart_df.set_index("Vibe"), height=320)

    st.divider()
    st.subheader(f"{top_emoji} Your playlist — {top_label} vibe")
    st.caption(f"{len(playlist)} tracks · ranked by image–song cosine similarity")

    has_spotify = "spotify_id" in playlist.columns

    for _, row in playlist.iterrows():
        with st.container():
            c1, c2 = st.columns([0.35, 0.65], gap="medium")

            with c1:
                # Spotify embed iframe — plays 30-sec previews directly in the app
                if has_spotify and row.get("spotify_id"):
                    st.components.v1.html(
                        f"""
                        <iframe
                            src="https://open.spotify.com/embed/track/{row['spotify_id']}?utm_source=generator&theme=0"
                            width="100%" height="80"
                            frameborder="0"
                            allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture"
                            loading="lazy">
                        </iframe>
                        """,
                        height=88,
                    )
                else:
                    st.caption("_Not found on Spotify_")

            with c2:
                st.markdown(f"**{row['track_name']}**")
                st.caption(f"{row['artists']}  ·  {row['genres']}")
                st.caption(f"similarity: {row['similarity']:.4f}")

        st.divider()


if __name__ == "__main__":
    main()
