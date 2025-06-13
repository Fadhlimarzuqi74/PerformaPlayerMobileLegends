import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import ast

# 1. Load Dataset
st.title("Fuzzy Logic Performa Player Mobile Legends")

@st.cache_data
def load_data():
    df = pd.read_csv('MPLID_S13_POS.csv')
    return df

df = load_data()

# 2. Pilih 5 Hero Favorit Per Role (Interactive)
st.header("1️⃣ Pilih 5 Hero Favorit per Role")
roles = df['Player_Role'].unique()
top_hero_per_role = {}

col1, col2 = st.columns(2)
for i, role in enumerate(roles):
    heroes = df[df['Player_Role'] == role]['Hero_Pick'].value_counts().index.tolist()
    with (col1 if i % 2 == 0 else col2):
        selected_heroes = st.multiselect(
            f"Pilih 5 hero favorit untuk role: {role}",
            heroes, 
            default=heroes[:5], 
            max_selections=5,
            key=f"fav_{role}"
        )
        top_hero_per_role[role] = selected_heroes

# 3. Filter Data Berdasarkan Hero Favorit
df_top = df[df.apply(lambda row: row['Hero_Pick'] in top_hero_per_role.get(row['Player_Role'], []), axis=1)]

st.write("Jumlah data setelah filter:", len(df_top))

# 4. Menentukan Batasan Fuzzy (min, mean, max)
st.header("2️⃣ Menentukan Batasan Fuzzy per Fitur")
stat_cols = ['KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret']

@st.cache_data
def get_fuzzy_limits(df, top_hero_per_role):
    limits_list = []
    for role, hero_list in top_hero_per_role.items():
        df_role = df[(df['Player_Role'] == role) & (df['Hero_Pick'].isin(hero_list))]
        for col in stat_cols:
            vals = df_role[col].dropna()
            if len(vals) == 0:
                continue
            min_val, mean_val, max_val = vals.min(), vals.mean(), vals.max()
            limits_list.append({
                'Role': role,
                'Variable': col,
                'min_val': min_val,
                'mean_val': mean_val,
                'max_val': max_val
            })
    return pd.DataFrame(limits_list)

fuzzy_limits = get_fuzzy_limits(df, top_hero_per_role)
st.dataframe(fuzzy_limits)

# 5. Visualisasi Batasan dalam Box Graph
st.header("3️⃣ Visualisasi Batasan Fuzzy (Box Graph)")

def box_graph(df):
    variables = stat_cols
    subplot_titles = variables
    fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=False, subplot_titles=subplot_titles)
    variable_colors = {
        "KDA": "#1f77b4",
        "Gold": "#ff7f0e",
        "Level": "#2ca02c",
        "Partisipation": "#d62728",
        "Damage_Dealt": "#9467bd",
        "Damage_Taken": "#8c564b",
        "Damage_Turret": "#e377c2"
    }
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

box_graph(fuzzy_limits)

# 6. Fuzzification
st.header("4️⃣ Fuzzification Nilai Statistik")

def fuzzify(min_val, mean_val, max_val, x):
    if x <= mean_val:
        mu_low = (mean_val - x) / (mean_val - min_val) if mean_val != min_val else 0
    else:
        mu_low = 0
    if x >= mean_val:
        mu_high = (x - mean_val) / (max_val - mean_val) if max_val != mean_val else 0
    else:
        mu_high = 0
    if min_val < x < mean_val:
        mu_med = (x - min_val) / (mean_val - min_val) if mean_val != min_val else 0
    elif mean_val < x < max_val:
        mu_med = (max_val - x) / (max_val - mean_val) if max_val != mean_val else 0
    elif x == mean_val:
        mu_med = 1
    else:
        mu_med = 0
    return [round(mu_low, 2), round(mu_med, 2), round(mu_high, 2)]

@st.cache_data
def calculateMuValues(df, fuzzy_limits, stat_cols):
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

all_players_fuzzy = calculateMuValues(df_top.copy(), fuzzy_limits, stat_cols)
st.dataframe(all_players_fuzzy.head(10))

# 7. Pembentukan Inferensi Fuzzy (Rule Base)
st.header("5️⃣ Pembentukan Rule Base Fuzzy")

def createInferenceTable_fuzzyLabels(fuzzy_limits, output_var='Performance', output_membership=None):
    fuzzy_labels = ['low', 'medium', 'high']
    all_combinations = list(itertools.product(fuzzy_labels, repeat=len(fuzzy_limits)))
    rules = []
    for comb in all_combinations:
        rule = dict(zip(fuzzy_limits, comb))
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

role_features = {
    'Jungler':     ['KDA', 'Level', 'Partisipation', 'Damage_Taken'],
    'Midlane':     ['KDA', 'Gold', 'Partisipation', 'Damage_Dealt'],
    'Explane':     ['KDA', 'Level', 'Partisipation', 'Damage_Taken'],
    'Goldlane':    ['KDA', 'Gold', 'Damage_Dealt', 'Damage_Turret'],
    'Roamer':      ['KDA', 'Gold', 'Partisipation', 'Damage_Taken'],
}

def getAllRules():
    all_rules = []
    for role, feats in role_features.items():
        rules = createInferenceTable_fuzzyLabels(feats, output_var='Performance', output_membership=['bad', 'decent', 'good'])
        rules.insert(0, 'Role', role)
        all_rules.append(rules)
    rules_df = pd.concat(all_rules, ignore_index=True)
    all_columns = ['Role','KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret', 'Performance']
    for col in all_columns:
        if col not in rules_df.columns:
            rules_df[col] = 'none'
    rules_df = rules_df[all_columns]
    rules_df = rules_df.fillna('none')
    return rules_df

all_rules_df = getAllRules()
st.dataframe(all_rules_df.head(10))

# 8. Sampling & Defuzzifikasi
st.header("6️⃣ Sampling & Defuzzifikasi")

def getRandomizedSample(fuzzy_limits, role, stat_cols, n_per_label=3):
    fuzzy_labels = ['low', 'medium', 'high']
    samples = {col: [] for col in stat_cols}
    samples['fuzzy_label'] = []
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

st.write("Contoh random sample untuk role Jungler")
jungler_sample = getRandomizedSample(fuzzy_limits, 'Jungler', role_features['Jungler'])
st.dataframe(jungler_sample)

# 9. Implementasi: Cari Hero & Hasil Fuzzy Logic
st.header("7️⃣ Implementasi: Cek Hasil Fuzzy Suatu Hero")

# Pilih Role, lalu Hero
chosen_role = st.selectbox("Pilih Role", roles)
heroes_for_role = df_top[df_top['Player_Role']==chosen_role]['Hero_Pick'].unique().tolist()
chosen_hero = st.selectbox("Pilih Hero", heroes_for_role)

# Cari baris pemain dengan hero & role tsb
hasil_fuzzy = all_players_fuzzy[(all_players_fuzzy['Player_Role']==chosen_role) & (all_players_fuzzy['Hero_Pick']==chosen_hero)]

def get_fuzzy_label(mu_str):
    try:
        if isinstance(mu_str, str):
            # Jika string, coba parse list/tuple dari string
            mu = ast.literal_eval(mu_str)
        else:
            # Jika sudah list/array
            mu = list(mu_str)
        idx = mu.index(max(mu))
        return ['low', 'medium', 'high'][idx]
    except Exception as e:
        # Jika gagal parsing, anggap tidak valid
        return "none"
        
rows = []
for idx, row in hasil_fuzzy.iterrows():
    features = role_features.get(chosen_role, [])
    label_row = {'Player_Name': row['Player_Name'], 'Player_Role': chosen_role, 'Hero_Pick': chosen_hero}
    for feat in features:
        mu_col = f'mu_{feat}'
        if mu_col in row and pd.notna(row[mu_col]):
            label_row[feat] = get_fuzzy_label(row[mu_col])
        else:
            label_row[feat] = 'none'
    # Cek rule
    cond = (all_rules_df['Role'] == chosen_role)
    for feat in features:
        cond &= (all_rules_df[feat] == label_row[feat])
    matched = all_rules_df[cond]
    performance = matched['Performance'].iloc[0] if not matched.empty else 'unknown'
    label_row['Performance'] = performance
    rows.append(label_row)

if rows:
    st.subheader("Hasil Fuzzy Logic untuk Hero & Role yang Dipilih:")
    st.table(pd.DataFrame(rows))
else:
    st.warning("Tidak ditemukan data untuk kombinasi hero/role tersebut.")

st.write("Selesai 🚀. Kembangkan fitur lebih lanjut sesuai kebutuhan.")
