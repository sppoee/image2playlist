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
├── src/
│   ├── embeddings.py       # CLIP encoding for images and songs
│   ├── recommender.py      # Cosine similarity + playlist generation
│   ├── data_utils.py       # Spotify dataset loading & preprocessing
│   └── description.py      # Convert audio features → text descriptions
├── data/
│   └── README.md           # Instructions for downloading Spotify dataset
├── notebooks/
│   ├── 01_exploration.ipynb     # Dataset EDA
│   ├── 02_embedding_viz.ipynb   # t-SNE / UMAP visualization
│   └── 03_evaluation.ipynb      # Evaluation metrics & user study results
├── demo/
│   └── app.py              # Gradio demo app
├── tests/
│   └── test_recommender.py
├── requirements.txt
└── README.md
```
