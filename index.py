import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import ast

@st.cache_data
def load_data():
    all_players = pd.read_csv('all_players.csv')
    fuzzy_limits = pd.read_csv('fuzzy_limits.csv')
    fuzzylogic_final = pd.read_csv('fuzzylogic_final.csv')
    rules = pd.read_csv('playerinference_rules.csv')
    fuzzified = pd.read_csv('fuzzified_player.csv')
    return all_players, fuzzy_limits, fuzzylogic_final, rules, fuzzified

all_players, fuzzy_limits, fuzzylogic_final, rules, fuzzified = load_data()

# --- SIDEBAR: tambah menu baru ---
st.sidebar.title("Fuzzy Logic Player Performance")
menu = st.sidebar.radio(
    "Pilih halaman:",
    (
        "Statistik Dataset",
        "Visualisasi Fuzzy Limits",
        "Tabel Inferensi (Rules)",  # <-- TAMBAHAN
        "Tabel Fuzzyfikasi",
        "Hasil Fuzzy Logic"
    )
)

# --- TABEL FUZZY LIMITS + VISUALISASI ---
if menu == "Visualisasi Fuzzy Limits":
    st.title("Tabel & Visualisasi Fuzzy Limits")

    st.subheader("Tabel Fuzzy Limits")
    st.caption("Min, Mean, Max setiap fitur statistik untuk masing-masing role (hasil dari fuzzy_limits.csv).")
    st.dataframe(fuzzy_limits)

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

# --- MENU BARU: TABEL INFERENSI (RULES) ---
if menu == "Tabel Inferensi (Rules)":
    st.title("Tabel Inferensi Fuzzy Logic (Rule Base)")
    st.caption("Aturan inferensi fuzzy logic per role berdasarkan kombinasi label (low/medium/high) masing-masing fitur.")
    st.dataframe(rules)
    st.info("Kolom Performance adalah output hasil inferensi fuzzy berdasarkan kombinasi nilai fuzzy setiap fitur.")

# --- TABEL FUZZYFIKASI ---
if menu == "Tabel Fuzzyfikasi":
    st.title("Tabel Hasil Fuzzyfikasi Derajat Keanggotaan")
    st.caption("Setiap fitur pemain dikonversi ke derajat keanggotaan fuzzy (low, medium, high) berbentuk list [μ_low, μ_med, μ_high].")
    pemain = st.selectbox("Pilih Nama Pemain", options=all_players['Player_Name'].unique())
    st.write(fuzzified[fuzzified['Player_Name'] == pemain])

# --- HASIL AKHIR FUZZY LOGIC ---
if menu == "Hasil Fuzzy Logic":
    st.title("Hasil Akhir Inferensi Fuzzy Logic")
    st.dataframe(fuzzylogic_final)
    st.subheader("Rekapitulasi Jumlah Performance per Role")
    summary = fuzzylogic_final.groupby(['Player_Role','Performance'])['Player_Name'].count().unstack().fillna(0).astype(int)
    st.dataframe(summary)
    st.subheader("Distribusi Performance")
    import plotly.express as px
    fig = px.histogram(fuzzylogic_final, x="Performance", color="Player_Role", barmode='group')
    st.plotly_chart(fig)
    st.info("Label Performance: **good**, **decent**, **bad** berdasarkan aturan fuzzy rule base.")

st.markdown("---")
st.caption("App dibuat dengan :heart: oleh [Fadhlimarzuqi74](https://github.com/Fadhlimarzuqi74)")
