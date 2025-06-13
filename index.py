import os
import requests
import json
import pandas as pd
from requests import get
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import itertools

st.set_page_config(layout="centered")
st.title('Mobile Legends:Bang-Bang Player Performance')
st.write("Program ini berfungsi untuk menentukan performa seorang pemain Mobile Legends:Bang-Bang berdasarkan hero yang dipilihnya dan posisi yang dimainkan. Sistem ini menggunakan dataset pertandingan MPL ID S13.")

# === LOAD DATASET DAN CEK FILE ===
def safe_read_csv(file, **kwargs):
    if not os.path.exists(file):
        st.warning(f"File `{file}` tidak ditemukan, pastikan file ini tersedia di direktori!")
        return None
    return pd.read_csv(file, **kwargs)

df = safe_read_csv('MPLID_S13_POS.csv')
if df is None:
    st.stop()

roles = df['Player_Role'].unique()

jungler_list = ['Fredrinn', 'Baxia', 'Ling', 'Barats', 'Akai']
explane_list = ['Cici', 'Terizla', 'Yu Zhong', 'Xborg', 'Masha']
midlane_list = ['Luo Yi', 'Valentina', 'Novaria', 'Faramis', 'Pharsa']
roamer_list = ['Ruby', 'Minotaur', 'Chip', 'Edith', 'Franco']
goldlane_list = ['Roger', 'Claude', 'Karrie', 'Natan', 'Moskov']

hero_populer = jungler_list + explane_list + midlane_list + roamer_list + goldlane_list

player_list = df[df['Hero_Pick'].isin(hero_populer)]['Player_Name'].unique().tolist()

def append_player_list(hero_list, df):
    return df[df['Hero_Pick'].isin(hero_list)]['Player_Name'].unique().tolist()

# --- Simpan data per role ---
def save_role_csv(df, role, hero_list, filename):
    role_df = df[(df['Player_Role'] == role) & (df['Hero_Pick'].isin(hero_list))]
    role_df.to_csv(filename, index=False)
    return role_df

jungler_players = save_role_csv(df, 'Jungler', jungler_list, 'jungler_players.csv')
explane_players = save_role_csv(df, 'Explane', explane_list, 'explane_players.csv')
midlane_players = save_role_csv(df, 'Midlane', midlane_list, 'midlane_players.csv')
roamer_players = save_role_csv(df, 'Roamer', roamer_list, 'roamer_players.csv')
goldlane_players = save_role_csv(df, 'Goldlane', goldlane_list, 'goldlane_players.csv')

all_players = pd.concat([
    jungler_players,
    midlane_players,
    roamer_players,
    explane_players,
    goldlane_players
], ignore_index=True)
all_players.to_csv('all_players.csv', index=False)

# --- HITUNG FUZZY LIMITS ---
def get_role_limits(hero_list, df_role):
    df = df_role[df_role['Hero_Pick'].isin(hero_list)]
    num_cols = ['KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret']
    limits = df[num_cols].agg(['min', 'mean', 'max']).T
    limits = limits.rename(columns={'min': 'min_val', 'mean': 'mean_val', 'max': 'max_val'})
    limits = limits.reset_index().rename(columns={'index': 'Variable'})
    return limits

def getAllFuzzyLimits():
    jungler_limits = get_role_limits(jungler_list, jungler_players)
    jungler_limits.insert(0, 'Role', 'Jungler')
    midlane_limits = get_role_limits(midlane_list, midlane_players)
    midlane_limits.insert(0, 'Role', 'Midlane')
    roamer_limits = get_role_limits(roamer_list, roamer_players)
    roamer_limits.insert(0, 'Role', 'Roamer')
    explane_limits = get_role_limits(explane_list, explane_players)
    explane_limits.insert(0, 'Role', 'Explane')
    goldlane_limits = get_role_limits(goldlane_list, goldlane_players)
    goldlane_limits.insert(0, 'Role', 'Goldlane')
    all_limits = pd.concat([jungler_limits, midlane_limits, roamer_limits, explane_limits, goldlane_limits], ignore_index=True)
    return all_limits

# --- FUZZY MEMBERSHIP ---
def fuzzify(min_val, mean_val, max_val, x):
    # LOW
    if x <= mean_val:
        mu_low = (mean_val - x) / (mean_val - min_val) if mean_val != min_val else 0
    else:
        mu_low = 0
    # HIGH
    if x >= mean_val:
        mu_high = (x - mean_val) / (max_val - mean_val) if max_val != mean_val else 0
    else:
        mu_high = 0
    # MEDIUM
    if min_val < x < mean_val:
        mu_med = (x - min_val) / (mean_val - min_val) if mean_val != min_val else 0
    elif mean_val < x < max_val:
        mu_med = (max_val - x) / (max_val - mean_val) if max_val != mean_val else 0
    elif x == mean_val:
        mu_med = 1
    else:
        mu_med = 0
    return [round(mu_low, 2), round(mu_med, 2), round(mu_high, 2)]

def calculateMuValues(df, fuzzy_limits, stat_cols=None):
    if stat_cols is None:
        stat_cols = ['KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret']
    for col in stat_cols:
        mu_list = []
        for idx, row in df.iterrows():
            role = row['Player_Role']
            fuzzy_row = fuzzy_limits[(fuzzy_limits['Role'] == role) & (fuzzy_limits['Variable'] == col)]
            if fuzzy_row.empty:
                mu_list.append([0,0,0])
                continue
            min_val = fuzzy_row['min_val'].values[0]
            mean_val = fuzzy_row['mean_val'].values[0]
            max_val = fuzzy_row['max_val'].values[0]
            mu = fuzzify(min_val, mean_val, max_val, row[col])
            mu_list.append(str(mu))
        df[f'mu_{col}'] = mu_list
    return df

# --- TAMPILAN GRAFIK BATASAN DATASET ---
variables = ["KDA", "Gold", "Level", "Partisipation", "Damage_Dealt", "Damage_Taken", "Damage_Turret"]
subplot_titles = [
    "KDA Distribution", "Gold Distribution", "Level Distribution", 
    "Participation Distribution", "Damage Dealt Distribution", 
    "Damage Taken Distribution", "Damage Turret Distribution"]
variable_colors = {
    "KDA": "#1f77b4",
    "Gold": "#ff7f0e",
    "Level": "#2ca02c",
    "Partisipation": "#d62728",
    "Damage_Dealt": "#9467bd",
    "Damage_Taken": "#8c564b",
    "Damage_Turret": "#e377c2"
}

def boxGraph(df):
    fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=False, subplot_titles=subplot_titles)
    for i, var in enumerate(variables):
        color = variable_colors.get(var, "#333333")
        for role in df['Role'].unique():
            vals = df[(df['Role'] == role) & (df['Variable'] == var)]
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

# --- SECTION 1: Tampilkan statistik ---
st.write('## Batasan Dataset')
fuzzy_limits = getAllFuzzyLimits()
st.dataframe(fuzzy_limits)
st.write('Distribusi statistik per role:')
boxGraph(fuzzy_limits)

# ---- HERO INPUT ----
st.subheader('Cari Performa Pemain dari Nama Hero')
hero_name = st.text_input('Masukkan Nama Hero (case sensitive, contoh: Fredrinn):')
filtered = all_players if not hero_name else all_players[all_players['Hero_Pick'] == hero_name]

if filtered.empty:
    st.info("Tidak ada data untuk Hero yang dimasukkan.")
else:
    # Tampilkan data ringkas
    st.write(f"Menampilkan {len(filtered)} data pemain untuk hero `{hero_name or '[ALL]'}`")
    st.dataframe(filtered[['Player_Name','Player_Role','Hero_Pick'] + features], hide_index=True)

    # FUZZY + INFERENSI
    fuzzy_rows = []
    fuzzyval_rows = []   # Untuk tabel hasil fuzzifikasi
    for idx, row in filtered.iterrows():
        role = row['Player_Role']
        feats = role_features.get(role, [])
        vals, mlabels, val_dict = {}, {}, {}
        for f in feats:
            lims = fuzzy_limits[(fuzzy_limits['Role'] == role) & (fuzzy_limits['Variable'] == f)]
            if lims.empty: continue
            mu = fuzzify(lims['min_val'].values[0], lims['mean_val'].values[0], lims['max_val'].values[0], row[f])
            vals[f] = mu
            val_dict[f] = mu  # simpan nilai fuzzy untuk tabel fuzzifikasi
            mlabels[f] = fuzzy_label(mu) # label (low, med, high)
        # Proses inferensi rules (ambil performa dari tabel rules_df)
        rule_match = rules_df[(rules_df['Role'] == role)]
        for f in feats:
            rule_match = rule_match[rule_match[f] == mlabels[f]]
        performance = rule_match['Performance'].values[0] if not rule_match.empty else 'unknown'
        # Tabel hasil klasifikasi akhir
        fuzzy_rows.append({
            'Player_Name': row['Player_Name'],
            'Role': role,
            'Hero': row['Hero_Pick'],
            **mlabels,
            'Performance': performance
        })
        # Tabel hasil fuzzifikasi (nilai keanggotaan)
        fuzzyval_row = {
            'Player_Name': row['Player_Name'],
            'Role': role,
            'Hero': row['Hero_Pick']
        }
        for f in feats:
            fuzzyval_row[f'{f}_fuzzy'] = vals[f]  # contoh: [mu_low, mu_med, mu_high]
            fuzzyval_row[f'{f}_label'] = mlabels[f]  # contoh: low/med/high
        fuzzyval_rows.append(fuzzyval_row)

    # Tabel hasil fuzzifikasi
    st.subheader('Hasil Fuzzifikasi (Nilai Keanggotaan & Label Fuzzy)')
    st.dataframe(pd.DataFrame(fuzzyval_rows), hide_index=True)

    # Tabel hasil akhir fuzzy logic
    st.subheader('Hasil Klasifikasi Performa (Fuzzy Logic)')
    st.dataframe(pd.DataFrame(fuzzy_rows), hide_index=True)
st.success("Script telah diperbaiki agar dapat dijalankan pada Streamlit! Pastikan semua file .csv tersedia di folder yang sama dengan script ini.")
