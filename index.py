import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ast

# ---------- DATA LOADING ----------
@st.cache_data
def load_data():
    all_players = pd.read_csv('all_players.csv')
    fuzzy_limits = pd.read_csv('fuzzy_limits.csv')
    all_players_fuzzy = pd.read_csv('all_players_fuzzy.csv')
    rules = pd.read_csv('playerinference_rules.csv')
    fuzzified = pd.read_csv('fuzzified_player.csv')
    fuzzylogic_final = pd.read_csv('fuzzylogic_final.csv')
    return all_players, fuzzy_limits, all_player_fuzzy, rules, fuzzified, fuzzylogic_final

all_players, fuzzy_limits, all_player_fuzzy, rules, fuzzified, fuzzylogic_final = load_data()

# ---------- SIDEBAR ----------
st.sidebar.title("Fuzzy Logic Mobile Legends Player Performance")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ("Statistik Dataset", "Batasan Fuzzy", "Fuzzifikasi", "Tabel Inferensi (Rules)", "Defuzzifikasi", "Hasil Fuzzy Logic")
)

# ---------- 1. STATISTIK DATASET ----------
if menu == "Statistik Dataset":
    st.title("Statistik Dataset Pemain MPL Indonesia Season 13")
    st.markdown("**Dataset sudah difilter berdasarkan 5 hero yang paling banyak dimainkan di setiap role**")

    # Tambahkan keterangan hero teratas per role
    st.write("**Jungler:** Fredrinn, Baxia, Ling, Barats, Akai")
    st.write("**Explane:** Cici, Terizla, Yu Zhong, Xborg, Masha")
    st.write("**Midlane:** Luo Yi, Valentina, Novaria, Faramis, Pharsa")
    st.write("**Roamer:** Ruby, Minotaur, Chip, Edith, Franco")
    st.write("**Goldlane:** Roger, Claude, Karrie, Natan, Moskov")

    st.write(all_players)

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
    st.caption("Min, Mean, Max setiap fitur statistik untuk masing-masing role (hasil dari fuzzy_limits.csv).")
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
if menu == "Fuzzyfikasi":
    st.title("Tabel Hasil Fuzzifikasi")
    st.caption("Setiap fitur pemain dikonversi ke derajat keanggotaan fuzzy (low, medium, high) berbentuk list [μ_low, μ_med, μ_high].")
    st.dataframe(all_players_fuzzy)

    st.info("""
    Fuzzifikasi adalah proses mengubah data statistik pemain, seperti KDA atau Gold, menjadi nilai derajat keanggotaan fuzzy (μ) pada kategori Rendah, Sedang, dan Tinggi. 
    Setiap nilai μ menunjukkan seberapa besar suatu statistik termasuk dalam kategori tersebut, sehingga analisis data menjadi lebih fleksibel.
    """)

# --- 4. TABEL INFERENSI (RULES) ---
if menu == "Tabel Inferensi (Rules)":
    st.title("Tabel Inferensi Fuzzy Logic (Rule Base)")
    st.markdown("**Aturan inferensi fuzzy logic per role berdasarkan kombinasi variabel dari dataset dan label (low/medium/high) masing-masing fitur.**")
    st.dataframe(rules)
    st.info("Kolom Performance adalah output hasil inferensi fuzzy berdasarkan kombinasi nilai fuzzy setiap fitur.")
    
# ---------- 5. DEFUZZIFIKASI ----------
if menu == "Defuzzifikasi":
    st.title("Tabel Hasil Defuzzifikasi")
    st.caption("Setiap fitur pemain dikonversi ke derajat keanggotaan fuzzy (low, medium, high) berbentuk list [μ_low, μ_med, μ_high].")
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
