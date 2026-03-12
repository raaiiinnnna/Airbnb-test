# 🏠 Airbnb Austin – Analisis, Clustering & Prediksi Harga

Dashboard interaktif Airbnb Austin Texas dengan **Clustering Fasilitas** dan **Prediksi Harga** menggunakan Machine Learning.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/USERNAME/REPO/main/app.py)

---

## 📁 Struktur File

```
repo/
├── app.py                               ← Dashboard Streamlit
├── airbnb_austin_colab.py               ← Kode Google Colab (EDA + ML)
├── petra_-_-_-_listings_austins.xlsx    ← Dataset (10.533 listings)
├── requirements.txt                     ← Dependensi Python
└── README.md
```

---

## 🔬 Alur Analisis

```
Dataset (10.533 listings, 79 kolom)
        │
        ▼
  Preprocessing
  (bedrooms ÷10, parse amenities, encode boolean)
        │
        ├──────────────────────────────────────┐
        ▼                                      ▼
CLUSTERING FASILITAS                   PREDIKSI HARGA
(K-Means, tanpa harga)                 (Random Forest / Gradient Boosting)
 Input: one-hot amenitas               Input: semua fitur + cluster_rank
 Output: label cluster                 Target: price per malam ($)
        │                                      │
        └──────────┬───────────────────────────┘
                   ▼
          Dashboard Streamlit
  (EDA · Cluster · Prediksi · Peta · Explorer)
```

---

## 🎯 Fitur Dashboard (5 Tab)

| Tab | Konten |
|-----|--------|
| 📊 Overview EDA | Distribusi harga, tipe room, dampak fasilitas |
| 🔵 Clustering Fasilitas | Elbow + Silhouette, PCA 2D, profil cluster |
| 💰 Prediksi Harga | Evaluasi model, feature importance, simulasi prediksi |
| 🗺️ Peta Interaktif | Peta sebaran listings & cluster |
| 🔎 Data Explorer | Filter, sort, download CSV |

---

## 🚀 Deploy ke Streamlit Cloud

1. Push semua file ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. **New app** → pilih repo → `main` → `app.py`
4. Klik **Deploy!**

---

## 📓 Google Colab

1. Buka [colab.research.google.com](https://colab.research.google.com)
2. Buat notebook baru → copy-paste isi `airbnb_austin_colab.py` cell per cell
3. Aktifkan **CARA C** di CELL 2, isi URL GitHub Anda
4. Jalankan dari atas ke bawah

**Output Colab:** 10 visualisasi + model `.pkl` + CSV + ZIP

---

## 🤖 Model Machine Learning

### Clustering (K-Means)
- **Input:** 40 fitur one-hot amenitas + bedrooms, bathrooms, accommodates, amenity_count
- **Optimasi:** Elbow Method + Silhouette Score
- **Output:** Label cluster (Ekonomis / Standar / Nyaman / Premium / Mewah)

### Prediksi Harga (Supervised)
- **Target:** `price` (harga per malam dalam $)
- **Fitur:** bedrooms, bathrooms, accommodates, rating, neighbourhood, room_type, amenitas, **cluster_rank**
- **Model:** Random Forest & Gradient Boosting (dipilih yang R² lebih tinggi)
- **Evaluasi:** MAE, RMSE, R², MAPE

---

*Built with Python · Streamlit · scikit-learn · Plotly*
