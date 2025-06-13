import pandas as pd
import numpy as np
import plotly.express as px

# Load dataset utama
df = pd.read_csv('MPLID_S13_POS.csv')

# Daftar hero populer per role (sesuai hasil eksplorasi Dataset.ipynb)
jungler_list = ['Fredrinn', 'Baxia', 'Ling', 'Barats', 'Akai']
explane_list = ['Cici', 'Terizla', 'Yu Zhong', 'Xborg', 'Masha']
midlane_list = ['Luo Yi', 'Valentina', 'Novaria', 'Faramis', 'Pharsa']
roamer_list = ['Ruby', 'Minotaur', 'Chip', 'Edith', 'Franco']
goldlane_list = ['Roger', 'Claude', 'Karrie', 'Natan', 'Moskov']

# Filter data per role-hero
jungler_df = df[(df['Player_Role'] == 'Jungler') & (df['Hero_Pick'].isin(jungler_list))]
explane_df = df[(df['Player_Role'] == 'Explane') & (df['Hero_Pick'].isin(explane_list))]
midlane_df = df[(df['Player_Role'] == 'Midlane') & (df['Hero_Pick'].isin(midlane_list))]
roamer_df = df[(df['Player_Role'] == 'Roamer') & (df['Hero_Pick'].isin(roamer_list))]
goldlane_df = df[(df['Player_Role'] == 'Goldlane') & (df['Hero_Pick'].isin(goldlane_list))]

# Gabungkan semua data hasil filter
all_players = pd.concat([jungler_df, explane_df, midlane_df, roamer_df, goldlane_df], ignore_index=True)
all_players.reset_index(drop=True, inplace=True)

# Fitur yang digunakan untuk fuzzy logic
features = ['KDA', 'Gold', 'Level', 'Partisipation', 'Damage_Dealt', 'Damage_Taken', 'Damage_Turret']

fuzzy_limits = []
for role in all_players['Player_Role'].unique():
    role_df = all_players[all_players['Player_Role'] == role]
    for feat in features:
        vals = role_df[feat].dropna()
        if len(vals) == 0:
            continue
        min_val = vals.min()
        mean_val = vals.mean()
        max_val = vals.max()
        fuzzy_limits.append({'Role': role, 'Variable': feat, 'min_val': min_val, 'mean_val': mean_val, 'max_val': max_val})
df_limits = pd.DataFrame(fuzzy_limits)
df_limits.to_csv('fuzzy_limits.csv', index=False)

def fuzzify(minv, meanv, maxv, x):
    # Segitiga: Low (min, min, mean), Medium (min, mean, max), High (mean, max, max)
    if x <= minv: return {'low': 1, 'medium': 0, 'high': 0}
    if x >= maxv: return {'low': 0, 'medium': 0, 'high': 1}
    # Low
    low = max((meanv - x) / (meanv - minv), 0) if x < meanv else 0
    # Medium
    if x < meanv:
        medium = (x - minv) / (meanv - minv)
    else:
        medium = (maxv - x) / (maxv - meanv)
    medium = max(medium, 0)
    # High
    high = max((x - meanv) / (maxv - meanv), 0) if x > meanv else 0
    return {'low': low, 'medium': medium, 'high': high}

fuzzified_rows = []
for idx, row in all_players.iterrows():
    role = row['Player_Role']
    fuzz_feats = {}
    for feat in features:
        try:
            lims = df_limits[(df_limits['Role'] == role) & (df_limits['Variable'] == feat)].iloc[0]
            mu = fuzzify(lims['min_val'], lims['mean_val'], lims['max_val'], row[feat])
            # Simpan membership degree
            for label in ['low', 'medium', 'high']:
                fuzz_feats[f"{feat}_{label}"] = mu[label]
        except Exception as e:
            # Jika data tidak ada
            for label in ['low', 'medium', 'high']:
                fuzz_feats[f"{feat}_{label}"] = np.nan
    fuzzified_rows.append({'Player_Name': row['Player_Name'],
                           'Player_Role': role,
                           'Hero_Pick': row['Hero_Pick'],
                           **fuzz_feats})
df_fuzzified = pd.DataFrame(fuzzified_rows)
df_fuzzified.to_csv('fuzzified_players.csv', index=False)

df_rules = pd.read_csv('playerinference_rules.csv')
# Format: kolom [Role, KDA, Gold, ... dst, Performance]
def get_fuzzy_label(val):
    # Ambil label fuzzy dengan membership terbesar
    return max(val, key=val.get)

final_rows = []
for idx, row in all_players.iterrows():
    role = row['Player_Role']
    # Label fuzzy tiap fitur
    labels = {}
    for feat in features:
        lims = df_limits[(df_limits['Role'] == role) & (df_limits['Variable'] == feat)].iloc[0]
        mu = fuzzify(lims['min_val'], lims['mean_val'], lims['max_val'], row[feat])
        labels[feat] = get_fuzzy_label(mu)
    # Cari rule yang cocok
    df_rule = df_rules[df_rules['Role'] == role].copy()
    for feat in features:
        df_rule = df_rule[df_rule[feat] == labels[feat]]
    if not df_rule.empty:
        performance = df_rule.iloc[0]['Performance']
    else:
        performance = 'unknown'
    final_rows.append({**row, **labels, 'Performance': performance})

df_final = pd.DataFrame(final_rows)
df_final.to_csv('fuzzylogic_final.csv', index=False)

import plotly.express as px

for col in features:
    fig = px.box(df_final, y=col, color="Player_Role", points="all", title=f"Boxplot {col} per Role")
    fig.show()

print("Fuzzy Limits:")
display(df_limits)
print("Fuzzified Players:")
display(df_fuzzified)
print("Rules:")
display(df_rules)
print("Final Fuzzylogic:")
display(df_final)

import streamlit as st

st.title("Fuzzy Logic Player Performance - Mobile Legends")

st.subheader("Fuzzy Limits")
st.dataframe(df_limits)
st.subheader("Fuzzified Players")
st.dataframe(df_fuzzified)
st.subheader("Rules")
st.dataframe(df_rules)
st.subheader("Final Fuzzy Logic Results")
st.dataframe(df_final)
