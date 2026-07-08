# ♻️ EcoLens — AI-Powered Waste Intelligence System

A production-grade, end-to-end **deep learning system** that classifies waste from images using **EfficientNet-B4**, explains model decisions with **Grad-CAM heatmaps**, and provides India-specific disposal guidance via a **RAG-powered chatbot** — deployed on Hugging Face Spaces.

## Live App

 **[https://huggingface.co/spaces/Isha-03/EcoLens](https://huggingface.co/spaces/Isha-03/EcoLens)**

## How It Works

This system uses a **3-layer pipeline**:

| Layer | Technology | Role |
|-------|-----------|------|
|  **Classification** | EfficientNet-B4 fine-tuned | Classifies waste image into 7 categories |
|  **Explainability** | Grad-CAM (built from scratch) | Highlights which pixels influenced the prediction |
|  **RAG Chatbot** | FAISS + Groq LLM (Llama3) | Retrieves relevant knowledge and generates disposal guidance |

##  Features

-  **Waste Classification** — 7 categories with 89.6% test accuracy
-  **Grad-CAM Heatmaps** — visual explanation of model attention, implemented from scratch
-  **Bin Guidance** — shows correct bin across Green, Blue, Black, Red, E-Waste, Kabadiwala
-  **RAG Chatbot** — FAISS vector search + Groq LLM for India-specific Q&A
-  **Confidence Scores** — top-5 predictions with probability bars
- 🇮🇳 **India-Specific** — kabadiwala integration, CPCB guidelines, Swachh Bharat

##  Model Performance

| Metric | Value |
|--------|-------|
| Test Accuracy | **89.6%** |
| Best Val Accuracy | **89.01%** |
| Training Images | **13,740** |
| Waste Categories | **7** |
| Training Epochs | **4** |
| Model Architecture | **EfficientNet-B4** |

### Per-Class Results

| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| Glass | 0.924 | 0.873 | 0.898 |
| Hazardous | 0.885 | 0.958 | **0.920** |
| Metal | 0.768 | 0.891 | 0.825 |
| Non-Recyclable Trash | 0.955 | 0.891 | **0.922** |
| Organic Biodegradable | 0.909 | **1.000** | **0.952** |
| Paper/Cardboard | 0.932 | 0.925 | **0.929** |
| Recyclable Plastic | 0.743 | 0.770 | 0.756 |

##  Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10 |
| **Deep Learning** | PyTorch, EfficientNet-B4 |
| **Computer Vision** | OpenCV, Albumentations |
| **Explainability** | Grad-CAM (custom implementation) |
| **NLP / Embeddings** | Sentence Transformers (all-MiniLM-L6-v2) |
| **Vector Search** | FAISS |
| **LLM** | Groq API (Llama3-8b) |
| **UI Framework** | Streamlit |
| **Deployment** | Hugging Face Spaces + Docker |
| **Training Data** | TrashNet + Kaggle Garbage Classification |

##  System Architecture

```
User uploads waste image
        ↓
┌─────────────────────────────────────┐
│         EcoLens Pipeline            │
│                                     │
│  EfficientNet-B4 → Classification   │
│  Grad-CAM        → Heatmap          │
│  Detected Class  → RAG Chatbot      │
│                                     │
│  FAISS Search → Knowledge Base      │
│  Groq LLM     → Disposal Guidance   │
└─────────────────────────────────────┘
        ↓
Classification + Heatmap + Bin Guide
        ↓
Chatbot answers disposal questions
```

##  Datasets Used

| Dataset | Size | Purpose |
|---------|------|---------|
| [TrashNet (Stanford)](https://github.com/garythung/trashnet) | 2,527 images | 6 base waste classes |
| [Kaggle Garbage Classification](https://www.kaggle.com/datasets/mostafaabla/garbage-classification) | 15,515 images | 12 diverse waste classes |
| **Combined (after merge)** | **13,740 images** | **7 unified classes** |
