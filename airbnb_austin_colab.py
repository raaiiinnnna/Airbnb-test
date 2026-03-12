# ╔══════════════════════════════════════════════════════════════════╗
# ║      🏠 AIRBNB AUSTIN – ANALISIS LENGKAP | GOOGLE COLAB         ║
# ║      EDA  ▸  Clustering Fasilitas  ▸  Prediksi Harga            ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# Dataset  : Airbnb Listings Austin Texas
# Baris    : 10.533 listings  |  Kolom: 79
# Target   : price (harga per malam, $)
# ─────────────────────────────────────────────────────────────────
# CARA PAKAI:
#   1. Buka colab.research.google.com
#   2. File → New Notebook
#   3. Copy-paste tiap CELL ke dalam cell di Colab
#   4. Jalankan dari atas ke bawah (Shift + Enter)
# ==================================================================


# ══════════════════════════════════════════════════════════════════
# CELL 1 │ INSTALL & IMPORT LIBRARY
# ══════════════════════════════════════════════════════════════════
# !pip install pandas numpy matplotlib seaborn plotly scikit-learn openpyxl requests

import ast, os, warnings, zipfile, pickle
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

warnings.filterwarnings("ignore")
plt.rcParams.update({'figure.figsize': (13, 6), 'font.size': 12})
sns.set_theme(style="whitegrid", palette="muted")
SEED = 42

print("✅ Semua library berhasil diimport!")


# ══════════════════════════════════════════════════════════════════
# CELL 2 │ MUAT DATA
# ══════════════════════════════════════════════════════════════════
# Pilih SALAH SATU cara di bawah, hapus tanda # pada cara yang dipilih

# ── CARA A: Upload langsung dari komputer ─────────────────────
# from google.colab import files
# uploaded = files.upload()
# df_raw = pd.read_excel(list(uploaded.keys())[0])

# ── CARA B: Dari Google Drive ─────────────────────────────────
# from google.colab import drive
# drive.mount('/content/drive')
# df_raw = pd.read_excel('/content/drive/MyDrive/petra_-_-_-_listings_austins.xlsx')

# ── CARA C: Dari GitHub (DIREKOMENDASIKAN) ────────────────────
import io, requests
GITHUB_RAW = "https://raw.githubusercontent.com/USERNAME/REPO/main/petra_-_-_-_listings_austins.xlsx"
# ⚠️ Ganti USERNAME dan REPO dengan nama akun & repo GitHub Anda
resp = requests.get(GITHUB_RAW)
df_raw = pd.read_excel(io.BytesIO(resp.content))

print(f"✅ Data berhasil dimuat!")
print(f"   Baris : {df_raw.shape[0]:,}")
print(f"   Kolom : {df_raw.shape[1]}")
print(f"\nPreview 5 baris pertama:")
df_raw[['name','price','room_type','bedrooms','accommodates',
        'neighbourhood_cleansed','review_scores_rating']].head()


# ══════════════════════════════════════════════════════════════════
# CELL 3 │ PREPROCESSING
# ══════════════════════════════════════════════════════════════════

# 40 fasilitas yang akan dijadikan fitur clustering & ML
TOP_AMENITIES = [
    'wifi', 'kitchen', 'air conditioning', 'heating', 'washer', 'dryer',
    'free parking on premises', 'free street parking', 'pool', 'hot tub',
    'gym', 'elevator', 'tv', 'dedicated workspace', 'self check-in',
    'pets allowed', 'smoking allowed', 'bathtub', 'dishwasher',
    'refrigerator', 'microwave', 'coffee maker', 'hair dryer', 'iron',
    'fire extinguisher', 'first aid kit', 'carbon monoxide alarm',
    'smoke alarm', 'patio or balcony', 'bbq grill', 'backyard',
    'outdoor furniture', 'long term stays allowed', 'bed linens',
    'extra pillows and blankets', 'shampoo', 'private entrance',
    'baby crib', 'high chair', 'luggage dropoff allowed',
]

def parse_amenities(x):
    try:
        return [a.lower() for a in ast.literal_eval(x)] if isinstance(x, str) else []
    except:
        return []

def safe_key(s):
    return s.replace(' ', '_').replace('/', '_').replace('-', '_')

def preprocess(df):
    df = df.copy()

    # Kolom tanggal
    for c in ['host_since', 'first_review', 'last_review', 'last_scraped']:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')

    # Price: buang baris tanpa harga & outlier ekstrem
    df = df.dropna(subset=['price'])
    df = df[df['price'].between(10, 1500)].copy()

    # Bedrooms: di data ini dikali 10 → bagi 10
    df['bedrooms_clean'] = (df['bedrooms'] / 10).clip(0, 20)

    # Bathrooms: ekstrak angka dari teks
    df['bathrooms_num'] = (
        df['bathrooms_text'].astype(str)
        .str.extract(r'(\d+\.?\d*)')[0].astype(float)
    )

    # Boolean t/f → 1/0
    for c in ['host_is_superhost','instant_bookable',
              'host_has_profile_pic','host_identity_verified','has_availability']:
        if c in df.columns:
            df[c] = df[c].map({'t':1,'f':0,True:1,False:0})

    # Host experience (tahun)
    df['host_years'] = (
        (pd.Timestamp.now() - df['host_since']).dt.days / 365
    ).clip(0, 30).round(1)

    # Parse amenities → list
    df['amenities_list'] = df['amenities'].apply(parse_amenities)
    df['amenity_count']  = df['amenities_list'].apply(len)

    # One-hot encoding amenitas
    for amen in TOP_AMENITIES:
        df[f'amen_{safe_key(amen)}'] = df['amenities_list'].apply(
            lambda lst: 1 if amen in lst else 0
        )

    # Kategori harga
    df['price_category'] = pd.cut(df['price'],
        bins=[0,75,150,300,1500],
        labels=['Budget (<$75)','Mid ($75-150)','Premium ($150-300)','Luxury (>$300)'])

    df['neighbourhood_cleansed'] = df['neighbourhood_cleansed'].astype(str)

    return df

df = preprocess(df_raw)

print(f"✅ Preprocessing selesai!")
print(f"   Shape  : {df.shape[0]:,} baris × {df.shape[1]} kolom")
print(f"   Harga  : ${df['price'].min():.0f} – ${df['price'].max():.0f}")
print(f"   Median : ${df['price'].median():.0f} / malam")


# ══════════════════════════════════════════════════════════════════
# CELL 4 │ STATISTIK DESKRIPTIF
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("📊 RINGKASAN DATASET AIRBNB AUSTIN")
print("=" * 60)
stats = [
    ("Total Listings",                  f"{len(df):,}"),
    ("Total Host Unik",                 f"{df['host_id'].nunique():,}"),
    ("Superhost",                       f"{int(df['host_is_superhost'].sum()):,} ({df['host_is_superhost'].mean()*100:.1f}%)"),
    ("Instant Bookable",                f"{int(df['instant_bookable'].sum()):,} ({df['instant_bookable'].mean()*100:.1f}%)"),
    ("Rata-rata Harga/Malam",           f"${df['price'].mean():.2f}"),
    ("Median Harga/Malam",              f"${df['price'].median():.2f}"),
    ("Rating Rata-rata",                f"{df['review_scores_rating'].mean():.2f} / 5"),
    ("Rata-rata Jumlah Fasilitas",      f"{df['amenity_count'].mean():.1f}"),
]
for label, val in stats:
    print(f"  {label:<35}: {val}")

print(f"\nTipe Room:")
print(df['room_type'].value_counts().to_string())
print(f"\nKategori Harga:")
print(df['price_category'].value_counts().sort_index().to_string())


# ══════════════════════════════════════════════════════════════════
# CELL 5 │ VIZ 1 – DISTRIBUSI HARGA
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].hist(df['price'], bins=60, color='#4C72B0', edgecolor='white', alpha=0.85)
axes[0].axvline(df['price'].median(), color='red', ls='--', lw=2,
                label=f"Median: ${df['price'].median():.0f}")
axes[0].axvline(df['price'].mean(), color='orange', ls='--', lw=2,
                label=f"Mean: ${df['price'].mean():.0f}")
axes[0].set_title('Distribusi Harga per Malam', fontweight='bold')
axes[0].set_xlabel('Harga ($)')
axes[0].set_ylabel('Jumlah Listings')
axes[0].legend()

room_order = df['room_type'].value_counts().index.tolist()
data_box   = [df[df['room_type']==r]['price'].values for r in room_order]
bp = axes[1].boxplot(data_box, labels=room_order, patch_artist=True,
                     medianprops=dict(color='red', lw=2))
for patch, col in zip(bp['boxes'], ['#4C72B0','#55A868','#DD8452','#C44E52']):
    patch.set_facecolor(col); patch.set_alpha(0.7)
axes[1].set_title('Distribusi Harga per Tipe Room', fontweight='bold')
axes[1].set_xlabel('Tipe Room'); axes[1].set_ylabel('Harga ($)')
axes[1].tick_params(axis='x', rotation=10)
plt.suptitle(''); plt.tight_layout()
plt.savefig('viz1_distribusi_harga.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════
# CELL 6 │ VIZ 2 – TOP 30 FASILITAS TERPOPULER
# ══════════════════════════════════════════════════════════════════

all_amen = [a for lst in df['amenities_list'] for a in lst]
top30 = pd.DataFrame(Counter(all_amen).most_common(30), columns=['amenity','count'])
top30['pct'] = (top30['count'] / len(df) * 100).round(1)

fig, ax = plt.subplots(figsize=(12, 10))
bars = ax.barh(top30['amenity'][::-1], top30['pct'][::-1],
               color='#4C72B0', edgecolor='white', alpha=0.85)
ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=9)
ax.set_title('Top 30 Fasilitas Terpopuler di Airbnb Austin', fontsize=14, fontweight='bold')
ax.set_xlabel('% Listings yang Memiliki Fasilitas Ini')
ax.set_xlim(0, 120)
plt.tight_layout()
plt.savefig('viz2_top_fasilitas.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════
# CELL 7 │ VIZ 3 – DAMPAK FASILITAS TERHADAP HARGA
# ══════════════════════════════════════════════════════════════════

premium_list = ['pool','hot tub','gym','bbq grill','dedicated workspace',
                'patio or balcony','bathtub','free parking on premises',
                'pets allowed','self check-in','washer','dryer',
                'elevator','air conditioning','long term stays allowed']
avg_all = df['price'].mean()
impact  = []
for a in premium_list:
    col = f"amen_{safe_key(a)}"
    if col in df.columns:
        avg_with = df[df[col]==1]['price'].mean()
        impact.append({'fasilitas': a, 'avg_harga': avg_with,
                       'selisih': avg_with - avg_all})
impact_df = pd.DataFrame(impact).sort_values('selisih')

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

colors_imp = ['#55A868' if x >= 0 else '#C44E52' for x in impact_df['selisih']]
bars = axes[0].barh(impact_df['fasilitas'], impact_df['selisih'],
                    color=colors_imp, edgecolor='white')
axes[0].axvline(0, color='black', lw=1)
axes[0].bar_label(bars, fmt='$%.0f', padding=3, fontsize=9)
axes[0].set_title('Selisih Harga vs Rata-rata\n(listing DENGAN fasilitas vs tanpa)', fontweight='bold')
axes[0].set_xlabel('Selisih Harga ($)')

samp = df.sample(min(2500, len(df)), random_state=SEED)
sc = axes[1].scatter(samp['amenity_count'], samp['price'],
                     alpha=0.35, c=samp['price'], cmap='YlOrRd', s=15)
plt.colorbar(sc, ax=axes[1], label='Harga ($)')
z = np.polyfit(samp['amenity_count'], samp['price'], 1)
xline = np.linspace(samp['amenity_count'].min(), samp['amenity_count'].max(), 100)
axes[1].plot(xline, np.poly1d(z)(xline), color='blue', lw=2, ls='--', label='Tren')
axes[1].set_title('Jumlah Fasilitas vs Harga per Malam', fontweight='bold')
axes[1].set_xlabel('Jumlah Fasilitas'); axes[1].set_ylabel('Harga ($)')
axes[1].legend()
plt.tight_layout()
plt.savefig('viz3_dampak_fasilitas.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════
# CELL 8 │ VIZ 4 – NEIGHBOURHOOD & RATING
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

nb = (df.groupby('neighbourhood_cleansed')['price']
      .agg(['mean','count']).reset_index()
      .query('count >= 20').nlargest(12,'mean'))
bars = axes[0].barh(nb['neighbourhood_cleansed'], nb['mean'],
                    color='#4C72B0', edgecolor='white')
axes[0].bar_label(bars, fmt='$%.0f', padding=3)
axes[0].invert_yaxis()
axes[0].set_title('Top 12 Area – Rata-rata Harga Tertinggi', fontweight='bold')
axes[0].set_xlabel('Rata-rata Harga ($)')

df['review_scores_rating'].dropna().hist(ax=axes[1], bins=30,
    color='#55A868', edgecolor='white', alpha=0.85)
axes[1].axvline(df['review_scores_rating'].mean(), color='red', ls='--', lw=2,
                label=f"Mean: {df['review_scores_rating'].mean():.2f}")
axes[1].set_title('Distribusi Rating Keseluruhan', fontweight='bold')
axes[1].set_xlabel('Rating (1–5)'); axes[1].set_ylabel('Jumlah Listings')
axes[1].legend()
plt.tight_layout()
plt.savefig('viz4_neighbourhood_rating.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════
# CELL 9 │ VIZ 5 – HEATMAP KORELASI
# ══════════════════════════════════════════════════════════════════

num_cols = ['price','bedrooms_clean','beds','accommodates','bathrooms_num',
            'amenity_count','number_of_reviews','review_scores_rating',
            'review_scores_cleanliness','review_scores_value',
            'availability_365','minimum_nights','host_years',
            'estimated_occupancy_l365d','estimated_revenue_l365d']
num_cols = [c for c in num_cols if c in df.columns]

corr = df[num_cols].corr()
plt.figure(figsize=(13, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, cbar_kws={'shrink':0.8})
plt.title('Heatmap Korelasi Antar Variabel Numerik', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('viz5_korelasi.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════
# CELL 10 │ CLUSTERING FASILITAS – ELBOW & SILHOUETTE
# ══════════════════════════════════════════════════════════════════
# Clustering TIDAK menggunakan harga. Hanya berdasarkan fasilitas
# & fitur struktural listing (kamar, kapasitas, dll)

print("🔄 Menyiapkan fitur clustering dari fasilitas...")

amen_cols   = [c for c in df.columns if c.startswith('amen_')]
struct_cols = ['bedrooms_clean','bathrooms_num','accommodates','amenity_count','instant_bookable']
struct_cols = [c for c in struct_cols if c in df.columns]

X_cl       = df[amen_cols + struct_cols].fillna(0)
scaler_cl  = StandardScaler()
X_scaled   = scaler_cl.fit_transform(X_cl)

print(f"   Total fitur clustering : {X_cl.shape[1]}")
print(f"   Fitur amenitas         : {len(amen_cols)}")
print(f"   Fitur struktural       : {len(struct_cols)}")

inertias, sil_scores, K_RANGE = [], [], range(2, 10)

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, lbl, sample_size=3000, random_state=SEED))
    print(f"  K={k} | Inertia={km.inertia_:>12,.0f} | Silhouette={sil_scores[-1]:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(list(K_RANGE), inertias, 'o-', color='#4C72B0', lw=2.5, ms=8)
axes[0].set_title('Elbow Method – Menentukan K Optimal', fontweight='bold')
axes[0].set_xlabel('Jumlah Cluster (K)'); axes[0].set_ylabel('Inertia (WCSS)')
axes[0].grid(True, alpha=0.3)

axes[1].plot(list(K_RANGE), sil_scores, 's-', color='#55A868', lw=2.5, ms=8)
axes[1].set_title('Silhouette Score per K', fontweight='bold')
axes[1].set_xlabel('Jumlah Cluster (K)'); axes[1].set_ylabel('Silhouette Score')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Menentukan Jumlah Cluster Optimal', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('viz6_elbow_silhouette.png', dpi=150, bbox_inches='tight')
plt.show()

BEST_K = list(K_RANGE)[sil_scores.index(max(sil_scores))]
print(f"\n✅ K optimal: {BEST_K}  (Silhouette = {max(sil_scores):.4f})")


# ══════════════════════════════════════════════════════════════════
# CELL 11 │ FIT CLUSTERING FINAL & PROFIL CLUSTER
# ══════════════════════════════════════════════════════════════════

# Ganti BEST_K jika ingin jumlah cluster berbeda, contoh: BEST_K = 4
km_final = KMeans(n_clusters=BEST_K, random_state=SEED, n_init=20)
df['cluster'] = km_final.fit_predict(X_scaled)

# Beri label berdasarkan urutan rata-rata harga (bukan fitur clustering!)
price_rank  = df.groupby('cluster')['price'].mean().rank().astype(int)
rank_labels = {0:'🟢 Ekonomis', 1:'🔵 Standar', 2:'🟠 Nyaman', 3:'🔴 Premium', 4:'⭐ Mewah'}
df['cluster_rank']  = df['cluster'].map(price_rank - 1)
df['cluster_label'] = df['cluster_rank'].map(
    {i: rank_labels.get(i, f'Cluster {i}') for i in range(BEST_K)}
)

print(f"✅ Clustering selesai dengan K={BEST_K}")
print("\n📊 Distribusi Cluster:")
print(df['cluster_label'].value_counts().to_string())

# Profil cluster
profile_agg = {
    'Jumlah'        : ('id',                    'count'),
    'Avg_Harga_$'   : ('price',                 'mean'),
    'Median_Harga_$': ('price',                 'median'),
    'Avg_Fasilitas' : ('amenity_count',          'mean'),
    'Avg_Kamar'     : ('bedrooms_clean',         'mean'),
    'Avg_Tamu'      : ('accommodates',           'mean'),
    'Avg_Rating'    : ('review_scores_rating',   'mean'),
    'Pct_Pool_%'    : ('amen_pool',              'mean'),
    'Pct_HotTub_%'  : ('amen_hot_tub',           'mean'),
    'Pct_Workspace_%':('amen_dedicated_workspace','mean'),
    'Pct_Parking_%' : ('amen_free_parking_on_premises','mean'),
    'Pct_Superhost_%':('host_is_superhost',      'mean'),
}
cluster_profile = df.groupby('cluster_label').agg(**profile_agg).round(2)
for c in [col for col in cluster_profile.columns if 'Pct' in col]:
    cluster_profile[c] = (cluster_profile[c]*100).round(1).astype(str)+'%'

print("\n" + "="*70)
print("📋 PROFIL TIAP CLUSTER FASILITAS (harga sebagai info, BUKAN input clustering)")
print("="*70)
print(cluster_profile.to_string())
cluster_profile.to_csv('profil_cluster.csv')
print("\n✅ Disimpan: profil_cluster.csv")


# ══════════════════════════════════════════════════════════════════
# CELL 12 │ VIZ 7 – VISUALISASI CLUSTER
# ══════════════════════════════════════════════════════════════════

pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X_scaled)
df['pca1'] = X_pca[:, 0]
df['pca2'] = X_pca[:, 1]
explained = pca.explained_variance_ratio_ * 100

PALETTE = ['#2ecc71','#3498db','#e67e22','#e74c3c','#9b59b6','#1abc9c','#f39c12']
cl_colors = {lbl: PALETTE[i] for i, lbl in enumerate(sorted(df['cluster_label'].unique()))}

fig, axes = plt.subplots(1, 3, figsize=(19, 6))

# PCA Scatter
for lbl, grp in df.groupby('cluster_label'):
    axes[0].scatter(grp['pca1'], grp['pca2'], c=cl_colors[lbl],
                    label=lbl, alpha=0.4, s=10)
axes[0].set_title(f'Scatter PCA 2D\n(Var explained: {explained.sum():.1f}%)', fontweight='bold')
axes[0].set_xlabel(f'PC1 ({explained[0]:.1f}%)')
axes[0].set_ylabel(f'PC2 ({explained[1]:.1f}%)')
axes[0].legend(fontsize=8, markerscale=2)

# Box harga per cluster
order = df.groupby('cluster_label')['price'].mean().sort_values().index.tolist()
data_box = [df[df['cluster_label']==l]['price'].values for l in order]
bp = axes[1].boxplot(data_box, labels=order, patch_artist=True,
                     medianprops=dict(color='black', lw=2))
for patch, col in zip(bp['boxes'], PALETTE):
    patch.set_facecolor(col); patch.set_alpha(0.7)
axes[1].set_title('Distribusi Harga per Cluster\n(harga sebagai informasi)', fontweight='bold')
axes[1].set_xlabel('Cluster'); axes[1].set_ylabel('Harga ($)')
axes[1].tick_params(axis='x', rotation=20)
plt.sca(axes[1]); plt.title(axes[1].get_title())

# Rata-rata fitur per cluster
cl_feat = df.groupby('cluster_label')[['amenity_count','bedrooms_clean',
                                       'accommodates','bathrooms_num']].mean()
cl_feat = cl_feat.reindex(order)
cl_feat.plot(kind='bar', ax=axes[2], edgecolor='white', alpha=0.85, width=0.7)
axes[2].set_title('Rata-rata Fitur per Cluster', fontweight='bold')
axes[2].set_xlabel('Cluster'); axes[2].set_ylabel('Nilai Rata-rata')
axes[2].tick_params(axis='x', rotation=20)
axes[2].legend(['Fasilitas','Kamar','Tamu','Kamar Mandi'], fontsize=8)

plt.suptitle('Visualisasi Cluster Fasilitas Airbnb Austin', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('viz7_cluster.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════
# CELL 13 │ VIZ 8 – PETA INTERAKTIF CLUSTER (PLOTLY)
# ══════════════════════════════════════════════════════════════════

fig_map = px.scatter_mapbox(
    df.sample(min(4000, len(df)), random_state=SEED),
    lat='latitude', lon='longitude',
    color='cluster_label',
    size='price', size_max=14,
    color_discrete_sequence=PALETTE,
    hover_name='name',
    hover_data={'price':True,'room_type':True,'cluster_label':True,
                'amenity_count':True,'review_scores_rating':True,
                'neighbourhood_cleansed':True},
    mapbox_style='open-street-map',
    zoom=10, height=650,
    title='🗺️ Peta Cluster Fasilitas – Airbnb Austin',
)
fig_map.update_layout(margin=dict(r=0, t=50, l=0, b=0))
fig_map.show()
print("✅ Peta interaktif ditampilkan!")


# ══════════════════════════════════════════════════════════════════
# CELL 14 │ PERSIAPAN FITUR PREDIKSI HARGA
# ══════════════════════════════════════════════════════════════════
# Prediksi harga menggunakan label cluster sebagai SALAH SATU fitur
# → cluster memperkaya model, bukan sebaliknya

print("🔄 Menyiapkan fitur untuk prediksi harga...")

le_room = LabelEncoder()
le_nb   = LabelEncoder()
df['room_type_enc']      = le_room.fit_transform(df['room_type'])
df['neighbourhood_enc']  = le_nb.fit_transform(df['neighbourhood_cleansed'])

FEATURE_COLS = (
    ['bedrooms_clean','bathrooms_num','accommodates','beds',
     'amenity_count','minimum_nights','availability_365',
     'number_of_reviews','review_scores_rating','review_scores_cleanliness',
     'review_scores_value','review_scores_location',
     'host_is_superhost','instant_bookable','host_years',
     'room_type_enc','neighbourhood_enc',
     'cluster_rank']          # ← label cluster digunakan sebagai fitur
    + amen_cols               # ← one-hot amenitas
)
FEATURE_COLS = [c for c in FEATURE_COLS if c in df.columns]

ml_df = df[FEATURE_COLS + ['price']].dropna(subset=['price']).copy()
ml_df[FEATURE_COLS] = ml_df[FEATURE_COLS].fillna(ml_df[FEATURE_COLS].median())

X = ml_df[FEATURE_COLS]
y = ml_df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED)

print(f"✅ Data siap untuk training:")
print(f"   Train  : {len(X_train):,} baris")
print(f"   Test   : {len(X_test):,} baris")
print(f"   Fitur  : {len(FEATURE_COLS)}")
print(f"           (termasuk cluster_rank & {len(amen_cols)} fitur amenitas)")


# ══════════════════════════════════════════════════════════════════
# CELL 15 │ TRAINING – RANDOM FOREST
# ══════════════════════════════════════════════════════════════════

print("🔄 Melatih Random Forest Regressor...")

rf = RandomForestRegressor(
    n_estimators=200, max_depth=15, min_samples_leaf=5,
    max_features='sqrt', random_state=SEED, n_jobs=-1
)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)

mae_rf   = mean_absolute_error(y_test, y_pred_rf)
rmse_rf  = np.sqrt(mean_squared_error(y_test, y_pred_rf))
r2_rf    = r2_score(y_test, y_pred_rf)
mape_rf  = np.mean(np.abs((y_test - y_pred_rf) / y_test.clip(lower=1))) * 100

print(f"\n{'='*50}")
print(f"📊 PERFORMA RANDOM FOREST")
print(f"{'='*50}")
print(f"  MAE   (Mean Absolute Error)  : ${mae_rf:.2f}")
print(f"  RMSE  (Root Mean Sq Error)   : ${rmse_rf:.2f}")
print(f"  R²                           : {r2_rf:.4f}  ({r2_rf*100:.1f}% variance explained)")
print(f"  MAPE  (Mean Abs % Error)     : {mape_rf:.1f}%")


# ══════════════════════════════════════════════════════════════════
# CELL 16 │ TRAINING – GRADIENT BOOSTING
# ══════════════════════════════════════════════════════════════════

print("🔄 Melatih Gradient Boosting Regressor...")

gb = GradientBoostingRegressor(
    n_estimators=200, learning_rate=0.08, max_depth=5,
    min_samples_leaf=5, subsample=0.8, random_state=SEED
)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

mae_gb   = mean_absolute_error(y_test, y_pred_gb)
rmse_gb  = np.sqrt(mean_squared_error(y_test, y_pred_gb))
r2_gb    = r2_score(y_test, y_pred_gb)
mape_gb  = np.mean(np.abs((y_test - y_pred_gb) / y_test.clip(lower=1))) * 100

print(f"\n{'='*50}")
print(f"📊 PERFORMA GRADIENT BOOSTING")
print(f"{'='*50}")
print(f"  MAE   : ${mae_gb:.2f}")
print(f"  RMSE  : ${rmse_gb:.2f}")
print(f"  R²    : {r2_gb:.4f}  ({r2_gb*100:.1f}%)")
print(f"  MAPE  : {mape_gb:.1f}%")


# ══════════════════════════════════════════════════════════════════
# CELL 17 │ PERBANDINGAN MODEL & PILIH TERBAIK
# ══════════════════════════════════════════════════════════════════

compare = pd.DataFrame({
    'Model'    : ['Random Forest', 'Gradient Boosting'],
    'MAE ($)'  : [round(mae_rf,2),  round(mae_gb,2)],
    'RMSE ($)' : [round(rmse_rf,2), round(rmse_gb,2)],
    'R² (%)'   : [round(r2_rf*100,2), round(r2_gb*100,2)],
    'MAPE (%)' : [round(mape_rf,1),  round(mape_gb,1)],
})
print("\n📋 PERBANDINGAN MODEL:")
print(compare.to_string(index=False))

best_model   = rf if r2_rf >= r2_gb else gb
best_name    = 'Random Forest' if r2_rf >= r2_gb else 'Gradient Boosting'
y_pred_best  = y_pred_rf if r2_rf >= r2_gb else y_pred_gb
print(f"\n🏆 Model terbaik: {best_name}  (R² = {max(r2_rf, r2_gb)*100:.1f}%)")


# ══════════════════════════════════════════════════════════════════
# CELL 18 │ VIZ 9 – EVALUASI MODEL
# ══════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(19, 5))

# Actual vs Predicted
max_val = min(float(y_test.max()), 800)
axes[0].scatter(y_test, y_pred_best, alpha=0.35, s=12, color='#4C72B0')
axes[0].plot([0, max_val], [0, max_val], 'r--', lw=2, label='Prediksi Sempurna')
axes[0].set_xlim(0, max_val); axes[0].set_ylim(0, max_val)
axes[0].set_title(f'Actual vs Predicted – {best_name}', fontweight='bold')
axes[0].set_xlabel('Harga Aktual ($)'); axes[0].set_ylabel('Harga Prediksi ($)')
axes[0].legend()

# Residual
residuals = y_test - y_pred_best
axes[1].hist(residuals, bins=60, color='#55A868', edgecolor='white', alpha=0.85)
axes[1].axvline(0, color='red', ls='--', lw=2, label=f"Mean: ${residuals.mean():.2f}")
axes[1].set_title('Distribusi Residual (Error)', fontweight='bold')
axes[1].set_xlabel('Residual (Aktual − Prediksi, $)'); axes[1].set_ylabel('Jumlah')
axes[1].legend()

# MAE comparison
bars = axes[2].bar(['Random\nForest', 'Gradient\nBoosting'],
                   [mae_rf, mae_gb], color=['#4C72B0','#DD8452'],
                   edgecolor='white', width=0.5)
axes[2].bar_label(bars, fmt='$%.2f', padding=3)
axes[2].set_title('Perbandingan MAE Antar Model', fontweight='bold')
axes[2].set_ylabel('Mean Absolute Error ($)')

plt.suptitle('Evaluasi Model Prediksi Harga', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('viz9_evaluasi_model.png', dpi=150, bbox_inches='tight')
plt.show()


# ══════════════════════════════════════════════════════════════════
# CELL 19 │ VIZ 10 – FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════

importance = pd.DataFrame({
    'feature':    FEATURE_COLS,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

fig, ax = plt.subplots(figsize=(12, 8))
bars = ax.barh(importance['feature'][::-1], importance['importance'][::-1],
               color='#4C72B0', edgecolor='white', alpha=0.85)
ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
ax.set_title(f'Top 20 Feature Importance – {best_name}', fontsize=13, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('viz10_feature_importance.png', dpi=150, bbox_inches='tight')
plt.show()

print("\n🔑 TOP 10 FITUR PALING BERPENGARUH:")
print(importance.head(10).to_string(index=False))


# ══════════════════════════════════════════════════════════════════
# CELL 20 │ SIMULASI PREDIKSI HARGA
# ══════════════════════════════════════════════════════════════════

print("=" * 55)
print("🧮 SIMULASI PREDIKSI HARGA LISTING BARU")
print("=" * 55)

example = {c: 0 for c in FEATURE_COLS}
example.update({
    'bedrooms_clean'              : 2,      # 2 kamar tidur
    'bathrooms_num'               : 1,      # 1 kamar mandi
    'accommodates'                : 4,      # muat 4 tamu
    'beds'                        : 2,      # 2 tempat tidur
    'amenity_count'               : 38,     # 38 fasilitas
    'minimum_nights'              : 2,
    'availability_365'            : 200,
    'number_of_reviews'           : 30,
    'review_scores_rating'        : 4.8,
    'review_scores_cleanliness'   : 4.9,
    'review_scores_value'         : 4.7,
    'review_scores_location'      : 4.8,
    'host_is_superhost'           : 1,
    'instant_bookable'            : 1,
    'host_years'                  : 4.0,
    'room_type_enc'               : 0,      # 0 = Entire home/apt
    'neighbourhood_enc'           : 5,
    'cluster_rank'                : 2,      # cluster menengah
    'amen_wifi'                   : 1,
    'amen_kitchen'                : 1,
    'amen_air_conditioning'       : 1,
    'amen_free_parking_on_premises': 1,
    'amen_dedicated_workspace'    : 1,
    'amen_washer'                 : 1,
    'amen_dryer'                  : 1,
    'amen_self_check_in'          : 1,
    'amen_pool'                   : 0,
    'amen_hot_tub'                : 0,
})

X_ex  = pd.DataFrame([example])[FEATURE_COLS].fillna(0)
pred  = best_model.predict(X_ex)[0]

print(f"\n  Kamar Tidur   : {example['bedrooms_clean']:.0f}")
print(f"  Kamar Mandi   : {example['bathrooms_num']:.0f}")
print(f"  Kapasitas Tamu: {example['accommodates']:.0f}")
print(f"  Fasilitas     : {example['amenity_count']:.0f}")
print(f"  Rating        : {example['review_scores_rating']:.1f} / 5")
print(f"  Superhost     : {'Ya' if example['host_is_superhost'] else 'Tidak'}")
print(f"  Cluster       : {example['cluster_rank']:.0f}")
print(f"\n{'─'*40}")
print(f"  💰 PREDIKSI HARGA/MALAM : ${pred:.2f}")
print(f"  📅 Estimasi per Bulan   : ${pred*30:.2f}")
print(f"  📅 Estimasi per Tahun   : ${pred*365:.2f}")
print(f"{'─'*40}")


# ══════════════════════════════════════════════════════════════════
# CELL 21 │ SIMPAN MODEL & SEMUA FILE
# ══════════════════════════════════════════════════════════════════

# Simpan model & objek pendukung
artifacts = {
    'model_prediksi_harga.pkl'   : best_model,
    'kmeans_model.pkl'           : km_final,
    'scaler_cluster.pkl'         : scaler_cl,
    'encoder_room.pkl'           : le_room,
    'encoder_neighbourhood.pkl'  : le_nb,
    'feature_cols.pkl'           : FEATURE_COLS,
    'amen_cols.pkl'              : amen_cols,
    'top_amenities.pkl'          : TOP_AMENITIES,
}
for fname, obj in artifacts.items():
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)

# Export data hasil ke CSV
df[['id','name','neighbourhood_cleansed','room_type','price',
    'bedrooms_clean','accommodates','amenity_count',
    'review_scores_rating','cluster_label','cluster_rank']].to_csv(
    'hasil_analisis_lengkap.csv', index=False)

# Kemas semua ke ZIP
all_files = [f for f in os.listdir('.') if f.endswith(('.png','.csv','.pkl'))]
with zipfile.ZipFile('hasil_airbnb_austin.zip', 'w') as zf:
    for f in all_files:
        zf.write(f)

print(f"✅ Semua model & file tersimpan!")
print(f"📦 Total {len(all_files)} file dikemas ke: hasil_airbnb_austin.zip")
print(f"\nFile yang disimpan:")
for f in sorted(all_files): print(f"  {f}")

# ── Download hasil (aktifkan di Colab)
# from google.colab import files
# files.download('hasil_airbnb_austin.zip')
