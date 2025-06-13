import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
from itertools import product

st.set_page_config(layout="centered")
st.title('Mobile Legends:Bang-Bang Player Performance')
st.write("Program ini berfungsi untuk menentukan performa seorang pemain Mobile Legends:Bang-Bang berdasarkan hero yang dipilihnya dan posisi yang dimainkan. Sistem ini menggunakan dataset pertandingan semi-professional/professional patch 1.8.78C dengan logika Fuzzy.")

# Load dataset
df = pd.read_csv('MPLID_S13_POS.csv')
roles = df['Player_Role'].unique()

# List 5 hero populer per role (bisa diganti dynamic, tapi hardcode dari notebook)
jungler_list = ['Fredrinn', 'Baxia', 'Ling', 'Barats', 'Akai']
explane_list = ['Cici', 'Terizla', 'Yu Zhong', 'Xborg', 'Masha']
midlane_list = ['Luo Yi', 'Valentina', 'Novaria', 'Faramis', 'Pharsa']
roamer_list = ['Ruby', 'Minotaur', 'Chip', 'Edith', 'Franco']
goldlane_list = ['Roger', 'Claude', 'Karrie', 'Natan', 'Moskov']
hero_populer = jungler_list + explane_list + midlane_list + roamer_list + goldlane_list

# Filter hanya pemain yang main 25 hero populer
all_players = df[df['Hero_Pick'].isin(hero_populer)].copy()

# ---- FUZZY LIMITS ----
def get_fuzzy_limits(df):
    result = []
    role_map = {
        'Jungler': jungler_list,
        'Explane': explane_list,
        'Midlane': midlane_list,
        'Roamer': roamer_list,
        'Goldlane': goldlane_list,
    }
    for role, hero_list in role_map.items():
        temp = df[(df['Player_Role'] == role) & (df['Hero_Pick'].isin(hero_list))]
        agg = temp[features].agg(['min', 'mean', 'max']).T
        agg = agg.rename(columns={'min':'min_val', 'mean':'mean_val', 'max':'max_val'})
        agg['Role'] = role
        agg['Variable'] = agg.index
        result.append(agg[['Role','Variable','min_val','mean_val','max_val']])
    return pd.concat(result, ignore_index=True)

fuzzy_limits = get_fuzzy_limits(all_players)
st.subheader('Batasan Fuzzy per Statistik dan Role')
st.dataframe(fuzzy_limits, hide_index=True)

# ---- FUZZY LIMITS ----
def get_fuzzy_limits(df):
    result = []
    role_map = {
        'Jungler': jungler_list,
        'Explane': explane_list,
        'Midlane': midlane_list,
        'Roamer': roamer_list,
        'Goldlane': goldlane_list,
    }
    for role, hero_list in role_map.items():
        temp = df[(df['Player_Role'] == role) & (df['Hero_Pick'].isin(hero_list))]
        agg = temp[features].agg(['min', 'mean', 'max']).T
        agg = agg.rename(columns={'min':'min_val', 'mean':'mean_val', 'max':'max_val'})
        agg['Role'] = role
        agg['Variable'] = agg.index
        result.append(agg[['Role','Variable','min_val','mean_val','max_val']])
    return pd.concat(result, ignore_index=True)

fuzzy_limits = get_fuzzy_limits(all_players)
st.subheader('Batasan Fuzzy per Statistik dan Role')
st.dataframe(fuzzy_limits, hide_index=True)

# --- PINDAHKAN KODE BOX PLOT KE SINI ---
st.subheader('Distribusi Data Statistik Pemain')
fig = make_subplots(
    rows=len(features), cols=1, shared_yaxes=False,
    subplot_titles=[f"{x} Distribution" for x in features]
)
for i, feat in enumerate(features, 1):
    fig.add_trace(go.Box(x=all_players[feat], name=feat, boxpoints='outliers'), row=i, col=1)
fig.update_layout(height=300*len(features), width=800, showlegend=False)
st.plotly_chart(fig, use_container_width=True)


# ---- FUZZY MEMBERSHIP FUNCTION ----
def fuzzify(min_val, mean_val, max_val, x):
    if mean_val == min_val: mean_val += 1e-6
    if max_val == mean_val: max_val += 1e-6
    # LOW
    mu_low = max(0, (mean_val - x) / (mean_val - min_val)) if x <= mean_val else 0
    # HIGH
    mu_high = max(0, (x - mean_val) / (max_val - mean_val)) if x >= mean_val else 0
    # MED
    if min_val < x < mean_val:
        mu_med = (x - min_val)/(mean_val-min_val)
    elif mean_val < x < max_val:
        mu_med = (max_val-x)/(max_val-mean_val)
    elif x == mean_val:
        mu_med = 1
    else:
        mu_med = 0
    return [round(mu_low,2), round(mu_med,2), round(mu_high,2)]

def fuzzy_label(mu):
    idx = np.argmax(mu)
    return ['low','medium','high'][idx]

# ---- RULES ----
role_features = {
    'Jungler':     ['KDA', 'Level', 'Partisipation', 'Damage_Taken'],
    'Midlane':     ['KDA', 'Gold', 'Partisipation', 'Damage_Dealt'],
    'Explane':     ['KDA', 'Level', 'Partisipation', 'Damage_Taken'],
    'Goldlane':    ['KDA', 'Gold', 'Damage_Dealt', 'Damage_Turret'],
    'Roamer':      ['KDA', 'Gold', 'Partisipation', 'Damage_Taken'],
}
def get_rules():
    rules = []
    for role, feats in role_features.items():
        for comb in product(['low','medium','high'], repeat=len(feats)):
            high = comb.count('high')
            med = comb.count('medium')
            if high >= 2:
                perf = 'good'
            elif med >= 2:
                perf = 'decent'
            else:
                perf = 'bad'
            rule = {'Role': role}
            for f,l in zip(feats,comb): rule[f] = l
            rule['Performance'] = perf
            rules.append(rule)
    return pd.DataFrame(rules)
rules_df = get_rules()

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
    for idx, row in filtered.iterrows():
        role = row['Player_Role']
        feats = role_features.get(role, [])
        vals, mlabels = {}, {}
        for f in feats:
            lims = fuzzy_limits[(fuzzy_limits['Role']==role)&(fuzzy_limits['Variable']==f)]
            if lims.empty: continue
            mu = fuzzify(lims['min_val'].values[0], lims['mean_val'].values[0], lims['max_val'].values[0], row[f])
            vals[f] = mu
            mlabels[f] = fuzzy_label(mu)
        rule_match = rules_df[(rules_df['Role']==role)]
        for f in feats:
            rule_match = rule_match[rule_match[f]==mlabels[f]]
        performance = rule_match['Performance'].values[0] if not rule_match.empty else 'unknown'
        fuzzy_rows.append({'Player_Name':row['Player_Name'], 'Role':role, 'Hero':row['Hero_Pick'], **mlabels, 'Performance':performance})
    st.subheader('Hasil Klasifikasi Performa (Fuzzy Logic)')
    st.dataframe(pd.DataFrame(fuzzy_rows), hide_index=True)

st.caption("© 2024 - Fuzzy Player Performance MLBB - Streamlit Demo")
