import requests
import json
import pandas as pd
from requests import get
import random
from itertools import product
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
import random
import ast
import asyncio

# Load dataset
df = pd.read_csv('MPLID_S13_POS.csv')

# Daftar semua role unik
roles = df['Player_Role'].unique()

# Untuk menyimpan hasil top 5 hero tiap role
top_hero_per_role = {}

for role in roles:
    # Filter role tertentu
    df_role = df[df['Player_Role'] == role]
    # Ambil 5 hero teratas berdasarkan jumlah kemunculan
    top_heroes = df_role['Hero_Pick'].value_counts().head(5)
    top_hero_per_role[role] = top_heroes

# Hasil value_counts top 5 hero per role, simpan sebagai list
jungler_list = ['Fredrinn', 'Baxia', 'Ling', 'Barats', 'Akai']
explane_list = ['Cici', 'Terizla', 'Yu Zhong', 'Xborg', 'Masha']
midlane_list = ['Luo Yi', 'Valentina', 'Novaria', 'Faramis', 'Pharsa']
roamer_list = ['Ruby', 'Minotaur', 'Chip', 'Edith', 'Franco']
goldlane_list = ['Roger', 'Claude', 'Karrie', 'Natan', 'Moskov']

# Misalkan sudah ada list 25 hero populer (gabungkan semua list role)
hero_populer = jungler_list + explane_list + midlane_list + roamer_list + goldlane_list

# Dapatkan unique player yang pernah memainkan salah satu dari 25 hero tersebut
player_list = df[df['Hero_Pick'].isin(hero_populer)]['Player_Name'].unique().tolist()

def append_player_list(hero_list, df):
    return df[df['Hero_Pick'].isin(hero_list)]['Player_Name'].unique().tolist()

df = pd.read_csv('MPLID_S13_POS.csv')

# List hero populer tiap role (isi sesuai hasil sebelumnya)
jungler_list = ['Fredrinn', 'Baxia', 'Ling', 'Barats', 'Akai']
explane_list = ['Cici', 'Terizla', 'Yu Zhong', 'Xborg', 'Masha']
midlane_list = ['Luo Yi', 'Valentina', 'Novaria', 'Faramis', 'Pharsa']
roamer_list = ['Ruby', 'Minotaur', 'Chip', 'Edith', 'Franco']
goldlane_list = ['Roger', 'Claude', 'Karrie', 'Natan', 'Moskov']

# Filtering ganda: role dan hero sesuai list populer
jungler_df = df[(df['Player_Role'] == 'Jungler') & (df['Hero_Pick'].isin(jungler_list))]
explane_df = df[(df['Player_Role'] == 'Explane') & (df['Hero_Pick'].isin(explane_list))]
midlane_df = df[(df['Player_Role'] == 'Midlane') & (df['Hero_Pick'].isin(midlane_list))]
roamer_df = df[(df['Player_Role'] == 'Roamer') & (df['Hero_Pick'].isin(roamer_list))]
goldlane_df = df[(df['Player_Role'] == 'Goldlane') & (df['Hero_Pick'].isin(goldlane_list))]

# Simpan ulang ke file CSV
jungler_df.to_csv('jungler_players.csv', index=False)
explane_df.to_csv('explane_players.csv', index=False)
midlane_df.to_csv('midlane_players.csv', index=False)
roamer_df.to_csv('roamer_players.csv', index=False)
goldlane_df.to_csv('goldlane_players.csv', index=False)

# Load semua file CSV hasil split per role
jungler_players = pd.read_csv('jungler_players.csv')
midlane_players = pd.read_csv('midlane_players.csv')
roamer_players = pd.read_csv('roamer_players.csv')
explane_players = pd.read_csv('explane_players.csv')
goldlane_players = pd.read_csv('goldlane_players.csv')

# Gabungkan semua dataframe menjadi satu
all_players = pd.concat([
    jungler_players,
    midlane_players,
    roamer_players,
    explane_players,
    goldlane_players
], ignore_index=True)


all_players.isnull().sum()

all_players.duplicated().sum()

# make a bar graph for each feature using plotly express to check the distribution of each feature
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# make a box graph
fig = make_subplots(rows=7, cols=1, shared_yaxes=True, subplot_titles=('KDA Distribution', 'Gold Distribution', 'Level Distribution', 'Partisipation','Damage Dealt Distribution', 'Damage Taken Distribution', 'Damage Turret Distribution'))
fig.add_trace(go.Box(x=all_players['KDA'], name='KDA'), row=1, col=1)
fig.add_trace(go.Box(x=all_players['Gold'], name='Gold'), row=2, col=1)
fig.add_trace(go.Box(x=all_players['Level'], name='Level'), row=3, col=1)
fig.add_trace(go.Box(x=all_players['Partisipation'], name='Participation'), row=4, col=1)
fig.add_trace(go.Box(x=all_players['Damage_Dealt'], name='Damage_Dealt'), row=5, col=1)
fig.add_trace(go.Box(x=all_players['Damage_Taken'], name='Damage_Taken'), row=6, col=1)
fig.add_trace(go.Box(x=all_players['Damage_Turret'], name='Damage_Turret'), row=7, col=1)

fig.update_layout(height=1200, width=800)
st.plotly_chart(fig)

all_players.describe()

def all_players(hero_list, df_role):
    # Filter sesuai hero_list (optional, data kamu mungkin sudah terfilter)
    df = df_role[df_role['Hero_Pick'].isin(hero_list)]
    # Ambil fitur numerik
    num_cols = ['KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret']
    # Hitung batas fuzzy: min, mean, max
    limits = df[num_cols].agg(['min', 'mean', 'max']).T
    limits = limits.rename(columns={'min': 'min_val', 'mean': 'mean_val', 'max': 'max_val'})
    # Reset index agar fitur jadi kolom
    limits = limits.reset_index().rename(columns={'index': 'Variable'})
    return limits

def getAllFuzzyLimits():
    jungler_limits = all_players(jungler_list, jungler_players)
    jungler_limits.insert(0, 'Role', 'Jungler')

    midlane_limits = all_players(midlane_list, midlane_players)
    midlane_limits.insert(0, 'Role', 'Midlane')

    roamer_limits = all_players(roamer_list, roamer_players)
    roamer_limits.insert(0, 'Role', 'Roamer')

    explane_limits = all_players(explane_list, explane_players)
    explane_limits.insert(0, 'Role', 'Explane')

    goldlane_limits = all_players(goldlane_list, goldlane_players)
    goldlane_limits.insert(0, 'Role', 'Goldlane')

    #combine all dataframe
    all_limits = pd.concat([jungler_limits, midlane_limits, roamer_limits,  explane_limits , goldlane_limits], ignore_index=True)
    return all_limits

# Contoh fungsi fuzzy membership (segitiga) dengan min, mean, max
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
    # MEDIUM (segitiga puncak di mean)
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
            # Ambil batas fuzzy dari fuzzy_limits.csv berdasarkan Role dan Variable
            fuzzy_row = fuzzy_limits[(fuzzy_limits['Role'] == role) & (fuzzy_limits['Variable'] == col)]
            if fuzzy_row.empty:
                mu_list.append([0,0,0]) # fallback
                continue
            min_val = fuzzy_row['min_val'].values[0]
            mean_val = fuzzy_row['mean_val'].values[0]
            max_val = fuzzy_row['max_val'].values[0]
            mu = fuzzify(min_val, mean_val, max_val, row[col])
            mu_list.append(str(mu))
        df[f'mu_{col}'] = mu_list
    return df

def createInferenceTable_fuzzyLabels(fuzzy_limits, output_var='Performance', output_membership=None):
    """
    Membuat inference table dari kombinasi label fuzzy.
    """
    fuzzy_labels = ['low', 'medium', 'high']
    all_combinations = list(itertools.product(fuzzy_labels, repeat=len(fuzzy_limits)))
    rules = []
    for comb in all_combinations:
        rule = dict(zip(fuzzy_limits, comb))
        # Output logic: custom sesuai kebutuhan
        high_count = comb.count('high')
        med_count = comb.count('medium')
        if high_count >= 2:
            rule[output_var] = output_membership[2] if output_membership else 'good'
        elif med_count >= 2:
            rule[output_var] = output_membership[1] if output_membership else 'decent'
        else:
            rule[output_var] = output_membership[0] if output_membership else 'bad'
        rules.append(rule)
    return pd.DataFrame(rules)

import pandas as pd

def getAllRules():
    # Jungler: KDA, level, Partisipation, Damage_Turret
    jungler_rules = createInferenceTable_fuzzyLabels(['KDA', 'Level', 'Partisipation', 'Damage_Taken'], output_var='Performance', output_membership=['bad', 'decent', 'good'])
    jungler_rules.insert(0, 'Role', 'Jungler')
    
    # Midlane: KDA, Gold, Partisipation, Damage_Dealt
    midlane_rules = createInferenceTable_fuzzyLabels(['KDA', 'Gold', 'Partisipation', 'Damage_Dealt'], output_var='Performance', output_membership=['bad', 'decent', 'good'])
    midlane_rules.insert(0, 'Role', 'Midlane')
    
    # Exp: KDA, Level,Partisipation, Damage_Taken
    exp_rules = createInferenceTable_fuzzyLabels(['KDA', 'Level','Partisipation', 'Damage_Taken'], output_var='Performance', output_membership=['bad', 'decent', 'good'])
    exp_rules.insert(0, 'Role', 'Explane')
    
    # Goldlane: KDA, Gold, Damage_Dealt, Damage_Dealt, Damage_Turret
    goldlane_rules = createInferenceTable_fuzzyLabels(['KDA', 'Gold', 'Damage_Dealt', 'Damage_Turret'], output_var='Performance', output_membership=['bad', 'decent', 'good'])
    goldlane_rules.insert(0, 'Role', 'Goldlane')
    
    # Roamer: KDA, Gold, Partisipation, Damage_Taken, Damage_Turret
    roamer_rules = createInferenceTable_fuzzyLabels(['KDA', 'Gold', 'Partisipation', 'Damage_Taken'], output_var='Performance', output_membership=['bad', 'decent', 'good'])
    roamer_rules.insert(0, 'Role', 'Roamer')
    
    # Gabungkan semua
    all_rules = pd.concat([jungler_rules, midlane_rules, exp_rules, goldlane_rules, roamer_rules], ignore_index=True)

    # Pastikan semua kolom dari seluruh fitur ada
    all_columns = ['Role','KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret', 'Performance']
    for col in all_columns:
        if col not in all_rules.columns:
            all_rules[col] = 'none'
    # Urutkan kolom sesuai all_columns
    all_rules = all_rules[all_columns]

    # Ganti NaN dengan "none"
    all_rules = all_rules.fillna('NaN')
    
    return all_rules

def getRandomizedSample(fuzzy_limits, role, stat_cols=None, n_per_label=3):
    """
    Membuat sample random per label fuzzy (low, medium, high) untuk setiap kolom statistik
    berdasarkan fuzzy_limits (min_val, mean_val, max_val) untuk role tertentu.
    Return: DataFrame dengan kolom sesuai stat_cols + 'fuzzy_label'.
    """
    if stat_cols is None:
        stat_cols = fuzzy_limits[fuzzy_limits['Role'] == role]['Variable'].tolist()
    fuzzy_labels = ['low', 'medium', 'high']
    samples = {col: [] for col in stat_cols}
    samples['fuzzy_label'] = []

    # Ambil limit untuk role ini sebagai dict
    limits = {}
    for col in stat_cols:
        row = fuzzy_limits[(fuzzy_limits['Role'] == role) & (fuzzy_limits['Variable'] == col)]
        if not row.empty:
            limits[col] = {
                'min': row['min_val'].values[0],
                'mean': row['mean_val'].values[0],
                'max': row['max_val'].values[0]
            }

    for label in fuzzy_labels:
        for _ in range(n_per_label):
            for col in stat_cols:
                lim = limits[col]
                if label == 'low':
                    v = np.random.uniform(lim['min'], lim['mean'])
                elif label == 'medium':
                    v = np.random.uniform(lim['min'], lim['max'])
                elif label == 'high':
                    v = np.random.uniform(lim['mean'], lim['max'])
                samples[col].append(v)
            samples['fuzzy_label'].append(label)
    return pd.DataFrame(samples)

def defuzzification(inferenced_table):
    """
    Melakukan proses defuzzifikasi pada tabel inference hasil fuzzy.
    - Mengelompokkan nilai min(mu) untuk masing-masing kategori output ('bad', 'decent', 'good').
    - Mengembalikan nilai maksimum dari tiap kelompok sebagai representasi output crisp.
    """
    bad, decent, good = [], [], []

    # Tentukan kolom mu (float/int)
    mu_cols = [col for col in inferenced_table.columns 
               if col not in ['Role', 'Performance'] and np.issubdtype(inferenced_table[col].dtype, np.number)]

    # Drop kolom yang seluruhnya NaN
    inferenced_table = inferenced_table.dropna(axis=1, how='all')

    for idx, row in inferenced_table.iterrows():
        perf = row['Performance']
        mu_values = row[mu_cols].values
        # Jika semua mu adalah nan, skip
        mu_values = mu_values[~np.isnan(mu_values)]
        if len(mu_values) == 0:
            continue
        min_mu = np.min(mu_values)
        if perf == 'bad':
            bad.append(min_mu)
        elif perf == 'decent':
            decent.append(min_mu)
        elif perf == 'good':
            good.append(min_mu)

    # Jika tidak ada anggota, hasilkan 0 agar tidak error
    max_bad = max(bad) if bad else 0
    max_decent = max(decent) if decent else 0
    max_good = max(good) if good else 0

    return [max_bad, max_decent, max_good]

# Pastikan fuzzy_limits sudah dibaca sebelumnya
fuzzy_limits = pd.read_csv('fuzzy_limits.csv')

# Contoh generate random sample untuk semua role utama
roles = ['Jungler', 'Midlane', 'Explane', 'Goldlane', 'Roamer']
random_samples = []

for role in roles:
    # Pilih fitur yang sesuai dengan role, misal:
    if role == 'Jungler':
        stat_cols = ['KDA', 'Level', 'Partisipation', 'Damage_Taken']
    elif role == 'Midlane':
        stat_cols = ['KDA', 'Gold', 'Partisipation', 'Damage_Dealt']
    elif role == 'Explane':
        stat_cols = ['KDA', 'Level', 'Partisipation', 'Damage_Taken']
    elif role == 'Goldlane':
        stat_cols = ['KDA', 'Gold', 'Damage_Dealt', 'Damage_Turret']
    elif role == 'Roamer':
        stat_cols = ['KDA', 'Gold', 'Partisipation', 'Damage_Taken']
    else:
        continue

    sample = getRandomizedSample(fuzzy_limits, role=role, stat_cols=stat_cols, n_per_label=3)
    sample['Role'] = role
    random_samples.append(sample)

# Gabungkan seluruh sample
randomsample = pd.concat(random_samples, ignore_index=True)

# Pastikan stat_cols sudah didefinisikan sesuai dengan pipeline kamu
stat_cols = ['KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret']

# Hitung derajat keanggotaan fuzzy (mu) untuk setiap fitur/statistik pemain
fuzzy_df = calculateMuValues(all_players.copy(), fuzzy_limits)

# Pilih kolom yang ingin ditampilkan
selected_cols = ["Player_Name", "Hero_Pick", "Player_Role"] + [f"mu_{col}" for col in stat_cols if f"mu_{col}" in fuzzy_df.columns]
hasil_fuzzy = fuzzy_df[selected_cols]

# Tampilkan hasil (untuk Jupyter Notebook, gunakan display; kalau script biasa, gunakan print atau hasil_fuzzy.head())
display(hasil_fuzzy)

# Load data
df_fuzzy = pd.read_csv('fuzzified_player.csv')
df_rules = pd.read_csv('playerinference_rules.csv')

role_features = {
    'Jungler':     ['KDA', 'Level', 'Partisipation', 'Damage_Taken'],
    'Midlane':     ['KDA', 'Gold', 'Partisipation', 'Damage_Dealt'],
    'Explane':     ['KDA', 'Level', 'Partisipation', 'Damage_Taken'],
    'Goldlane':    ['KDA', 'Gold', 'Damage_Dealt', 'Damage_Turret'],
    'Roamer':      ['KDA', 'Gold', 'Partisipation', 'Damage_Taken'],
}

def get_fuzzy_label(mu_str):
    # Mendukung format string list atau string tuple, misal: '[0.2, 0.5, 0.3]'
    if isinstance(mu_str, str):
        mu = ast.literal_eval(mu_str)
    else:
        mu = list(mu_str)
    idx = mu.index(max(mu))
    return ['low', 'medium', 'high'][idx]

rows = []
for idx, row in df_fuzzy.iterrows():
    role = row['Player_Role']
    features = role_features.get(role, [])
    if not features:
        continue

    label_row = {
        'Player_Name': row['Player_Name'],
        'Player_Role': role,
        'Hero_Pick': row['Hero_Pick'],
    }
    # Konversi: kolom fitur berisi label fuzzy
    for feat in features:
        mu_col = f'mu_{feat}'
        if mu_col in row and pd.notna(row[mu_col]):
            label_row[feat] = get_fuzzy_label(row[mu_col])
        else:
            label_row[feat] = 'none'

    # Cek rule
    cond = (df_rules['Role'] == role)
    for feat in features:
        cond &= (df_rules[feat] == label_row[feat])
    matched = df_rules[cond]
    performance = matched['Performance'].iloc[0] if not matched.empty else 'unknown'
    label_row['Performance'] = performance

    rows.append(label_row)

# Buat urutan kolom yang diinginkan
final_columns = ['Player_Name', 'Player_Role', 'Hero_Pick']
# Tambahkan fitur sesuai role, lalu 'Performance'
for role, feats in role_features.items():
    for feat in feats:
        if feat not in final_columns:
            final_columns.append(feat)
final_columns.append('Performance')

# Buang duplikat urutan fitur
final_columns = final_columns[:3] + list(dict.fromkeys(final_columns[3:-1])) + [final_columns[-1]]

df_final = pd.DataFrame(rows)
df_final = df_final[[col for col in final_columns if col in df_final.columns]]

st.write('# Mobile Legends:Bang-Bang Player Performance')
st.write('Program ini berfungsi untuk menentukan performa seorang pemain Mobile Legends:Bang-Bang berdasarkan hero yang dipilihnya dan posisi yang dimainkan')
st.write('Sistem ini menggunakan dataset yang diambil dari pertandingan semi-professional dan professional Mobile Legends:Bang-Bang yang terjadi di patch 1.8.78C')
st.write('Menggunakan Fuzzy Logic, dataset diubah menjadi batasan-batasan fuzzy yang kemudian digunakan untuk memprediksi performa pemain')

st.write('\n\n')

st.write('## Batasan Dataset')
st.write('Berikut adalah batasan dataset yang digunakan')
fuzzy_limits = pd.read_csv('fuzzy_limits.csv')
boxGraph(fuzzy_limits)
fuzzy_limits = getAllFuzzyLimits() 
st.write(fuzzy_limits)

st.write('## Masukkan Nama Hero')
st.write('Nama Hero digunakan untuk mencari detail match yang dimainkan oleh player')
chosen = st.text_input('Nama Hero')
