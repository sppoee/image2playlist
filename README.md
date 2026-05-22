# 🎵 Image-to-Playlist Generation

> Upload an image, get a playlist that matches its vibe — powered by CLIP multimodal embeddings.

![Python](https://img.shields.io/badge/Python-3.9+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 📌 Overview

This project maps images to music playlists using **CLIP (Contrastive Language-Image Pretraining)**. By encoding both images and song descriptions into the same embedding space, we can find songs whose "vibe" matches a given photo.

```
User uploads image → CLIP Image Encoder → image embedding
                                                ↓
                                       Cosine Similarity
                                                ↓
Song descriptions → CLIP Text Encoder → song embeddings
                                                ↓
                                        Top-K playlist 🎶
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
│   └── 03_vibe_mapping.ipynb       # Vibe taxonomy, image encoder, playlist generation
├── .gitignore
└── README.md
```

> **Note:** `song_embeddings.npy` is excluded from git (>100MB). Run `02_clip_setup.ipynb` locally to generate it.

---

## 👥 Contributors

- Anisa Dye
- Sharon Lee
- Szuyu Chi
