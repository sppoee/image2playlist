# 🎵 image2playlist

> Upload an image, get a playlist that matches its vibe — powered by CLIP multimodal embeddings.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Overview

This project maps images to music playlists using **CLIP (Contrastive Language-Image Pretraining)**. By encoding both images and song descriptions into the same 512-dimensional embedding space, we can find songs whose "vibe" matches a given photo — no additional model training required.

```
User uploads image → CLIP Image Encoder (ViT-B/32) → image embedding [512d]
                                                              ↓
                                                    Cosine Similarity
                                                              ↓
Song descriptions → CLIP Text Encoder → song embeddings [61670, 512]
                                                              ↓
                                                  Vibe detection (12 categories)
                                                              ↓
                                                   Top-K diverse playlist 🎶
```

---

## 🗂 Project Structure

```
image2playlist/
├── data/
│   ├── spotify_cleaned_final.csv   # Cleaned Spotify dataset (61,670 tracks)
│   ├── song_index.csv              # Row-aligned song metadata for embeddings
│   ├── song_embeddings.npy         # Precomputed CLIP text embeddings [generated locally]
│   └── sample_images/              # Test images for pipeline evaluation
├── notebooks/
│   ├── 01_data_cleaning.ipynb      # Data ingestion, cleaning, clip_metadata construction
│   ├── 02_clip_setup.ipynb         # CLIP setup, song text embedding precomputation
│   └── 03_vibe_mapping.ipynb       # Vibe taxonomy, image encoder, playlist generation & eval
├── src/
│   └── helpers.py                  # Shared utilities: VIBE_TAXONOMY, build_playlist, Spotify helpers
├── demo/
│   └── app.py                      # Streamlit demo app
├── .env                            # Spotify API credentials (not committed)
├── .gitignore
└── README.md
```

> **Note:** `song_embeddings.npy` is excluded from git (>100MB). Run `02_clip_setup.ipynb` locally to generate it.

---

## 🎨 Vibe Taxonomy

The pipeline classifies images into one of **12 scene-based vibe categories**. Each vibe is defined by a natural-language probe text that CLIP encodes into a 512-d vector. Vibe detection works by comparing the image embedding to all 12 probe embeddings and selecting the closest match.

| Vibe | Emoji | Vibe | Emoji |
|------|-------|------|-------|
| Cozy Cafe | ☕️ | Morning / Calm | 🌅 |
| Winter / Snow | ❄️ | Dark / Moody | 🖤 |
| Beach / Summer | 🏖️ | Festival / Concert | 🎸 |
| Rainy / Melancholic | 🌧️ | Late Night Drive | 🌃 |
| Party / Energetic | 🎉 | Urban / Hustle | 🏙️ |
| Romantic Evening | 🕯️ | Nature / Peaceful | 🌿 |

Vibes and their probes are defined in `src/helpers.py` — edits there are automatically reflected in both the notebooks and the Streamlit app.

---

## 🚀 Running Locally

### 1. Install dependencies

```bash
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git ftfy regex tqdm
pip install streamlit spotipy pandas numpy pillow pillow-heif python-dotenv
```

### 2. Set up Spotify credentials (optional)

Create a `.env` file in the repo root:

```
SPOTIPY_CLIENT_ID=your_client_id
SPOTIPY_CLIENT_SECRET=your_client_secret
```

Without these, the app still works — Spotify embeds and 30-second previews just won't be available.

### 3. Generate song embeddings

Run `notebooks/02_clip_setup.ipynb` end-to-end. This produces:
- `data/song_embeddings.npy` — the full `[61670, 512]` embedding matrix
- `data/song_index.csv` — row-aligned song metadata

### 4. Launch the app

```bash
streamlit run demo/app.py
```

---


## 👥 Contributors

- Anisa Dye
- Sharon Lee
- Szuyu Chi

