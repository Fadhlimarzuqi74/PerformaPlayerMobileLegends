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
    fuzzylogic_final = pd.read_csv('fuzzylogic_final.csv')
    return all_players, fuzzy_limits, fuzzylogic_final

all_players, fuzzy_limits, fuzzylogic_final = load_data()

# ---------- SIDEBAR ----------
st.sidebar.title("Fuzzy Logic Player Performance")
menu = st.sidebar.radio(
    "Pilih halaman:",
    ("Statistik Dataset", "Visualisasi Fuzzy Limits", "Tabel Fuzzyfikasi", "Hasil Fuzzy Logic")
)

# ---------- 1. STATISTIK DATASET ----------
if menu == "Statistik Dataset":
    st.title("Statistik Dataset Pemain MPL")
    st.markdown("**Data 250 baris, 25 hero populer, 5 role utama**")
    st.write(all_players)

    st.subheader("Deskripsi Statistik")
    st.dataframe(all_players.describe())

    st.subheader("Cek Null & Duplicate")
    st.write("Missing values per kolom:")
    st.write(all_players.isnull().sum())
    st.write("Jumlah duplikasi:", all_players.duplicated().sum())

    st.info("Dataset sudah bersih dan siap digunakan.")

# ---------- 2. VISUALISASI FUZZY LIMITS ----------

# Tambahan: tampilkan tabel
    st.subheader("Tabel Batasan Fuzzy")
    st.caption("Min, Mean, Max setiap variabel untuk masing-masing role (hasil dari fuzzy_limits.csv).")
    st.dataframe(fuzzy_limits)
    
if menu == "Visualisasi Batasan Fuzzy":
    st.title("Visualisasi Distribusi Statistik Pemain (Boxplot per Fitur)")
    features = ['KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret']
    fig = make_subplots(
        rows=len(features), cols=1, shared_yaxes=False,
        subplot_titles=[f"{x} Distribution" for x in features]
    )
    for i, feat in enumerate(features, 1):
        fig.add_trace(go.Box(x=all_players[feat], name=feat, boxpoints='outliers'), row=i, col=1)
    fig.update_layout(height=300*len(features), width=800, showlegend=True)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Tabel Fuzzy Limits")
    st.caption("Min, Mean, Max setiap fitur statistik untuk masing-masing role (hasil dari fuzzy_limits.csv).")
    st.dataframe(fuzzy_limits)

    
# ---------- 3. TABEL FUZZYFIKASI ----------
if menu == "Tabel Fuzzyfikasi":
    st.title("Tabel Hasil Fuzzyfikasi Derajat Keanggotaan")
    st.caption("Setiap fitur pemain dikonversi ke derajat keanggotaan fuzzy (low, medium, high) berbentuk list [μ_low, μ_med, μ_high].")

    # Pilihan filter
    pemain = st.selectbox("Pilih Nama Pemain", options=all_players['Player_Name'].unique())
    df_fuzzy = pd.read_csv('fuzzified_player.csv')
    st.write(df_fuzzy[df_fuzzy['Player_Name'] == pemain])

    st.markdown("> **Keterangan:** Nilai pada kolom mu_xxx menunjukkan derajat keanggotaan fuzzy setelah proses fuzzification.")

# ---------- 4. HASIL FUZZY LOGIC ----------
if menu == "Hasil Fuzzy Logic":
    st.title("Hasil Akhir Inferensi Fuzzy Logic")
    st.markdown("Tiap baris adalah hasil inferensi fuzzy logic untuk **setiap pemain** (role, hero, fuzzy label fitur, hasil performance).")
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
