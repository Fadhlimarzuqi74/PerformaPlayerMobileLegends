import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ast

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    df = pd.read_csv('MPLID_S13_POS.csv')
    all_players = pd.read_csv('all_players.csv')
    fuzzy_limits = pd.read_csv('fuzzy_limits.csv')
    all_players_fuzzy = pd.read_csv('all_players_fuzzy.csv')
    rules = pd.read_csv('playerinference_rules.csv')
    fuzzified = pd.read_csv('fuzzified_player.csv')
    fuzzylogic_final = pd.read_csv('fuzzylogic_final.csv')
    return df, all_players, fuzzy_limits, all_players_fuzzy, rules, fuzzified, fuzzylogic_final

df, all_players, fuzzy_limits, all_players_fuzzy, rules, fuzzified, fuzzylogic_final = load_data()

# ---------- SIDEBAR ----------
st.sidebar.title("Fuzzy Logic Mobile Legends Player Performance")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ("Statistik Dataset", "Batasan Fuzzy", "Fuzzifikasi", "Tabel Inferensi (Rules)", "Defuzzifikasi", "Hasil Fuzzy Logic")
)

# ---------- 1. STATISTIK DATASET ----------
if menu == "Statistik Dataset":
    st.title("Statistik Dataset Pemain MPL Indonesia")
    st.write("Sistem ini menggunakan dataset yang diambil dari pertandingan profesional Mobile Legends Season 13 pada babak Play Off yang terjadi di patch 1.8.78C")
    st.dataframe(df)
            
    st.markdown("**Dataset sudah difilter berdasarkan 5 hero yang paling banyak dimainkan di setiap role**")

    # Tambahkan keterangan hero teratas per role
    st.write("**Jungler:** Fredrinn, Baxia, Ling, Barats, Akai")
    st.write("**Explane:** Cici, Terizla, Yu Zhong, Xborg, Masha")
    st.write("**Midlane:** Luo Yi, Valentina, Novaria, Faramis, Pharsa")
    st.write("**Roamer:** Ruby, Minotaur, Chip, Edith, Franco")
    st.write("**Goldlane:** Roger, Claude, Karrie, Natan, Moskov")

    st.dataframe(all_players)

    st.subheader("Cek Null & Duplicate")
    st.write("Missing values per kolom:")
    st.write(all_players.isnull().sum())
    st.write("Jumlah duplikasi:", all_players.duplicated().sum())

    st.info("Dataset sudah bersih dan siap digunakan.")

# ---------- 2. VISUALISASI FUZZY LIMITS ----------

if menu == "Batasan Fuzzy":
    st.title("Tabel & Visualisasi Batasan Fuzzy")

    # --- Tampilkan tabel lebih dulu ---
    st.subheader("Batasan Fuzzy")
    st.write("""Tabel ini menunjukkan batas nilai minimum, rata-rata, dan maksimum setiap variabel dataset 
    untuk setiap role. Batas ini digunakan sebagai acuan pembuatan fungsi keanggotaan fuzzy.""")
    st.dataframe(fuzzy_limits)

    # --- Lanjut visualisasi di bawahnya ---
    st.subheader("Visualisasi Distribusi Statistik Pemain (Boxplot per Fitur)")
    features = ['KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret']
    fig = make_subplots(
        rows=len(features), cols=1, shared_yaxes=False,
        subplot_titles=[f"{x} Distribution" for x in features]
    )
    for i, feat in enumerate(features, 1):
        fig.add_trace(go.Box(x=all_players[feat], name=feat, boxpoints='outliers'), row=i, col=1)
    fig.update_layout(height=300*len(features), width=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

# ---------- 3. TABEL FUZZIFIKASI ----------
if menu == "Fuzzifikasi":
    st.title("Tabel Hasil Fuzzifikasi")
    st.write("""Setiap variabel statistik pemain diubah ke derajat keanggotaan fuzzy (μ): Rendah, Sedang, Tinggi.""")
    st.dataframe(all_players_fuzzy)

    st.write("**Contoh Hasil Fuzzyfikasi (Player: Reyy, Role: Jungler, Hero: Ling):**")
    st.write("- **mu_KDA:** `[0, 0.35, 0.65]` → KDA pemain ini 35% Medium, 65% High")
    st.write("- **mu_Gold:** `[0, 0.91, 0.09]` → Gold yang didapat 91% Medium, 9% High")
    st.write("- **mu_Level:** `[0, 0, 1.0]` → Level 100% High")
    st.write("- **mu_Partisipation:** `[0, 0.78, 0.22]` → Partisipasi 78% Medium, 22% High")
    st.write("- **mu_Damage_Dealt:** `[0.01, 0.99, 0]` → Damage Dealt 99% Medium")
    st.write("- **mu_Damage_Taken:** `[0.98, 0.02, 0]` → Damage Taken 98% Low")
    st.write("- **mu_Damage_Turret:** `[0, 0.47, 0.53]` → Damage Turret 47% Medium, 53% High")

# --- 4. TABEL INFERENSI (RULES) ---
if menu == "Tabel Inferensi (Rules)":
    st.title("Tabel Inferensi Fuzzy Logic (Rule Base)")
    st.write("""
Tabel inferensi (rules) fuzzy logic pada aplikasi ini dibentuk secara otomatis berdasarkan kombinasi seluruh label fuzzy (low, medium, high) untuk variabel-variabel utama pada tiap role.

- Untuk setiap role (Jungler, Midlane, Explane, Goldlane, Roamer), dipilih 4 variabel statistik yang paling relevan sebagai input fuzzy.
    - Jungler : KDA, level, Partisipation, Damage_Taken
    - Midlane : KDA, Gold, Partisipation, Damage_Dealt
    - Explane : KDA, Level,Partisipation, Damage_Taken
    - Goldlane: KDA, Gold, Damage_Dealt, Damage_Turret
    - Roamer  : KDA, Level, Partisipation, Damage_Taken
    
- Semua kombinasi label fuzzy dari fitur-fitur tersebut di-generate, lalu ditentukan output performa ('Performance') menggunakan aturan:
    - Jika kombinasi mengandung **2 atau lebih 'high'**, maka performa = **good**
    - Jika kombinasi mengandung **2 atau lebih 'medium'** (dan kurang dari 2 'high'), maka performa = **decent**
    - Sisanya, performa = **bad**
- Hasil akhirnya berupa tabel yang memetakan setiap kombinasi label fuzzy ke nilai performa (bad/decent/good) untuk masing-masing role.
""")
    st.info("Kolom Performance adalah output hasil inferensi fuzzy berdasarkan kombinasi nilai fuzzy setiap fitur.")
    st.dataframe(rules)
    
    
# ---------- 5. DEFUZZIFIKASI ----------
if menu == "Defuzzifikasi":
    st.title("Tabel Hasil Defuzzifikasi")
    st.write("""
    Melakukan proses defuzzifikasi pada tabel inference hasil fuzzy.
    - Mengelompokkan nilai min(mu) untuk masing-masing kategori output ('bad', 'decent', 'good').
    - Mengembalikan nilai maksimum dari tiap kelompok sebagai representasi output crisp.
    """)
    st.dataframe(fuzzified)

    st.markdown("> **Keterangan:** Nilai pada kolom mu_xxx menunjukkan derajat keanggotaan fuzzy setelah proses fuzzification.")

# ---------- 5. HASIL FUZZY LOGIC ----------
if menu == "Hasil Fuzzy Logic":
    st.title("Hasil Akhir Inferensi Fuzzy Logic")

    # Pilih pemain untuk filter hasil tabel
    pemain = st.selectbox("Pilih Nama Pemain", options=fuzzylogic_final['Player_Name'].unique())
    filtered = fuzzylogic_final[fuzzylogic_final['Player_Name'] == pemain]
    st.dataframe(filtered)

    with st.expander("Lihat Seluruh Hasil Tabel (semua pemain)"):
        st.dataframe(fuzzylogic_final)

    st.subheader("Rekapitulasi Jumlah Performance per Role")
    summary = fuzzylogic_final.groupby(['Player_Role','Performance'])['Player_Name'].count().unstack().fillna(0).astype(int)
    st.dataframe(summary)

    st.subheader("Distribusi Performance")
    import plotly.express as px
    fig = px.histogram(fuzzylogic_final, x="Performance", color="Player_Role", barmode='group')
    st.plotly_chart(fig)
    st.info("Label Performance: **good**, **decent**, **bad** berdasarkan aturan fuzzy rule base.")
    
# ---------- FOOTER ----------
st.markdown("---")
st.caption("App dibuat dengan :heart: oleh [Fadhlimarzuqi74](https://github.com/Fadhlimarzuqi74)")
