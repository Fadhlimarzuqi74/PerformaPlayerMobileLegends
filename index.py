import streamlit as st
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import itertools

# --- Load Data ---
df = pd.read_csv('all_players.csv')
fuzzy_limits = pd.read_csv('fuzzy_limits.csv')

# --- Membership Function ---
def fuzzify_value(val, min_val, mean_val, max_val):
    # Triangular membership: low, medium, high
    if val <= min_val:
        return {'low': 1, 'medium': 0, 'high': 0}
    elif val < mean_val:
        low = (mean_val - val) / (mean_val - min_val)
        med = (val - min_val) / (mean_val - min_val)
        return {'low': low, 'medium': med, 'high': 0}
    elif val == mean_val:
        return {'low': 0, 'medium': 1, 'high': 0}
    elif val < max_val:
        med = (max_val - val) / (max_val - mean_val)
        high = (val - mean_val) / (max_val - mean_val)
        return {'low': 0, 'medium': med, 'high': high}
    else:
        return {'low': 0, 'medium': 0, 'high': 1}

# --- Inference Table Function (from your rule generator) ---
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

# --- Streamlit UI ---
st.title("Fuzzy Logic Player Performance - Mobile Legends MPL-ID")

# Penjelasan dan batasan dataset
st.header("Penjelasan Dataset & Batasan")
st.markdown("""
Dataset berisi performa pemain di MPL-ID berdasarkan beberapa fitur statistik utama:
- **KDA, Gold, Level, Partisipation, Damage Dealt, Damage Taken, Damage Turret**
Batasan: Dataset hanya mencakup 5 hero terpopuler per role (Jungler, Midlane, Goldlane, Roamer, Explane).
""")

# Pilih Hero
hero_names = sorted(df['Hero_Pick'].unique())
selected_hero = st.selectbox("Pilih nama hero:", hero_names)

# Tampilkan data hero
hero_data = df[df['Hero_Pick'] == selected_hero]
if hero_data.empty:
    st.warning("Tidak ada data untuk hero yang dipilih.")
    st.stop()

st.subheader(f"Statistik Data untuk Hero: {selected_hero}")
st.dataframe(hero_data)

# Visualisasi Box Graph per Role & Variabel
st.header("Visualisasi Box Graph (Fuzzy Limits)")
variables = ["KDA", "Gold", "Level", "Partisipation", "Damage_Dealt", "Damage_Taken", "Damage_Turret"]
subplot_titles = [f"{v} Distribution" for v in variables]
variable_colors = {
    "KDA": "#1f77b4", "Gold": "#ff7f0e", "Level": "#2ca02c", "Partisipation": "#d62728",
    "Damage_Dealt": "#9467bd", "Damage_Taken": "#8c564b", "Damage_Turret": "#e377c2"
}

fig = make_subplots(rows=len(variables), cols=1, shared_xaxes=False, subplot_titles=subplot_titles)
for i, var in enumerate(variables):
    color = variable_colors.get(var, "#333333")
    for role in fuzzy_limits['Role'].unique():
        vals = fuzzy_limits[(fuzzy_limits['Role'] == role) & (fuzzy_limits['Variable'] == var)]
        if not vals.empty:
            row = vals.iloc[0]
            values = [row['min_val'], row['mean_val'], row['max_val']]
            fig.add_trace(go.Box(
                y=values, name=f"{role}", boxpoints='all', jitter=0.5,
                marker_color=color, showlegend=(i==0)
            ), row=i+1, col=1)
fig.update_layout(height=1800, width=800, showlegend=True)
st.plotly_chart(fig)

# Tampilkan Inference Table (Rules)
st.header("Inference Table (Rules Fuzzy)")
fuzzy_vars = variables  # Atur jika ingin pakai subset variabel
rules_df = createInferenceTable_fuzzyLabels(fuzzy_vars)
st.dataframe(rules_df)

# Fuzzifikasi dan Inferensi untuk Hero Terpilih
st.header("Fuzzifikasi dan Hasil Fuzzy Logic")

# Ambil satu baris data hero untuk contoh
sample = hero_data.iloc[0]
role = sample['Player_Role']
fuzzified = {}
for var in variables:
    # Ambil batas fuzzy untuk role & variabel
    limit = fuzzy_limits[(fuzzy_limits['Role'] == role) & (fuzzy_limits['Variable'] == var)]
    if not limit.empty:
        min_v, mean_v, max_v = limit.iloc[0][['min_val', 'mean_val', 'max_val']]
        fuzzified[var] = fuzzify_value(sample[var], min_v, mean_v, max_v)
    else:
        fuzzified[var] = {'low': 0, 'medium': 0, 'high': 0}

st.subheader("Nilai Keanggotaan Fuzzy (Fuzzifikasi)")
fuzzy_df = pd.DataFrame({k: v for k, v in fuzzified.items()}).T
st.dataframe(fuzzy_df)

# Tentukan label fuzzy (low/medium/high) dengan nilai keanggotaan tertinggi
fuzzy_labels = {k: max(v, key=v.get) for k, v in fuzzified.items()}
st.write("Label Fuzzy untuk tiap variabel:")
st.json(fuzzy_labels)

# Inferensi: cari rule yang cocok di rules_df
match_rule = rules_df
for var in fuzzy_vars:
    match_rule = match_rule[match_rule[var] == fuzzy_labels[var]]
hasil_fuzzy = match_rule['Performance'].values[0] if not match_rule.empty else "Tidak diketahui"

st.success(f"Hasil Fuzzy Logic untuk hero **{selected_hero}**: **{hasil_fuzzy.upper()}**")

st.markdown("""
---
**Note:**  
- Fuzzifikasi menggunakan batas min, mean, max per role & variabel.
- Inference table otomatis dengan aturan: >=2 high → good, >=2 medium → decent, sisanya bad (bisa diubah sesuai kebutuhan Anda).
""")
