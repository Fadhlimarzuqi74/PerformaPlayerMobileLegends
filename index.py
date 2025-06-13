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
    st.write(all_players.head())

    st.subheader("Deskripsi Statistik")
    st.dataframe(all_players.describe())

    st.subheader("Cek Null & Duplicate")
    st.write("Missing values per kolom:")
    st.write(all_players.isnull().sum())
    st.write("Jumlah duplikasi:", all_players.duplicated().sum())

    st.info("Dataset sudah bersih dan siap digunakan.")

# ---------- 2. VISUALISASI FUZZY LIMITS ----------
if menu == "Visualisasi Fuzzy Limits":
    st.title("Visualisasi Fuzzy Limits per Role & Variable")
    variables = ["KDA", "Gold", "Level", "Partisipation", "Damage_Dealt", "Damage_Taken", "Damage_Turret"]
    subplot_titles = [
        "KDA Distribution", "Gold Distribution", "Level Distribution", 
        "Participation Distribution", "Damage Dealt Distribution", 
        "Damage Taken Distribution", "Damage Turret Distribution"
    ]
    variable_colors = {
        "KDA": "#1f77b4",
        "Gold": "#ff7f0e",
        "Level": "#2ca02c",
        "Partisipation": "#d62728",
        "Damage_Dealt": "#9467bd",
        "Damage_Taken": "#8c564b",
        "Damage_Turret": "#e377c2"
    }
    fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=False, subplot_titles=subplot_titles)
    for i, var in enumerate(variables):
        color = variable_colors.get(var, "#333333")
        for role in fuzzy_limits['Role'].unique():
            vals = fuzzy_limits[(fuzzy_limits['Role'] == role) & (fuzzy_limits['Variable'] == var)]
            values = []
            if not vals.empty:
                row = vals.iloc[0]
                values = [row['min_val'], row['mean_val'], row['max_val']]
            if values:
                fig.add_trace(go.Box(
                    y=values, 
                    name=role, 
                    boxpoints='all', 
                    jitter=0.5,
                    marker_color=color,
                    showlegend=(i==0)
                ), row=i+1, col=1)
    fig.update_layout(height=1800, width=800, showlegend=True)
    st.plotly_chart(fig)

    st.markdown("> **Keterangan:** Titik-titik pada box plot adalah batas min, mean, dan max dari masing-masing variabel fuzzy untuk setiap role.")

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
