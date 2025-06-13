import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import numpy as np
import itertools

# Set page configuration
st.set_page_config(page_title="MPLID S13 Fuzzy Logic Analysis", layout="wide")

# Load data
@st.cache_data
def load_data():
    all_players = pd.read_csv('all_players.csv')
    fuzzy_limits = pd.read_csv('fuzzy_limits.csv')
    df_fuzzy = pd.read_csv('fuzzified_player.csv')
    df_rules = pd.read_csv('playerinference_rules.csv')
    df_final = pd.read_csv('fuzzylogic_final.csv')
    return all_players, fuzzy_limits, df_fuzzy, df_rules, df_final

all_players, fuzzy_limits, df_fuzzy, df_rules, df_final = load_data()

# Define variables and subplot titles for box graph
variables = ["KDA", "Gold", "Level", "Partisipation", "Damage_Dealt", "Damage_Taken", "Damage_Turret"]
subplot_titles = [
    "KDA Distribution", "Gold Distribution", "Level Distribution",
    "Participation Distribution", "Damage Dealt Distribution",
    "Damage Taken Distribution", "Damage Turret Distribution"
]

# Dictionary of colors for each variable
variable_colors = {
    "KDA": "#1f77b4",
    "Gold": "#ff7f0e",
    "Level": "#2ca02c",
    "Partisipation": "#d62728",
    "Damage_Dealt": "#9467bd",
    "Damage_Taken": "#8c564b",
    "Damage_Turret": "#e377c2"
}

# Box graph function
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
                    showlegend=(i == 0)
                ), row=i + 1, col=1)
    fig.update_layout(height=1800, width=800, showlegend=True, title_text="Fuzzy Limits Distribution by Role")
    return fig

# Fuzzy logic functions
def get_fuzzy_label(mu_str):
    if isinstance(mu_str, str):
        mu = ast.literal_eval(mu_str)
    else:
        mu = list(mu_str)
    idx = mu.index(max(mu))
    return ['low', 'medium', 'high'][idx]

role_features = {
    'Jungler': ['KDA', 'Level', 'Partisipation', 'Damage_Taken'],
    'Midlane': ['KDA', 'Gold', 'Partisipation', 'Damage_Dealt'],
    'Explane': ['KDA', 'Level', 'Partisipation', 'Damage_Taken'],
    'Goldlane': ['KDA', 'Gold', 'Damage_Dealt', 'Damage_Turret'],
    'Roamer': ['KDA', 'Gold', 'Partisipation', 'Damage_Taken'],
}

def get_performance_for_hero(hero_name):
    # Filter df_final for the selected hero
    hero_data = df_final[df_final['Hero_Pick'] == hero_name]
    if hero_data.empty:
        return pd.DataFrame(), f"No data found for hero: {hero_name}"
    
    return hero_data, f"Performance results for hero: {hero_name}"

# Streamlit app layout
st.title("MPLID S13 Fuzzy Logic Performance Analysis")

# Sidebar for hero selection
st.sidebar.header("Hero Selection")
hero_list = df_final['Hero_Pick'].unique().tolist()
selected_hero = st.sidebar.selectbox("Select a Hero", hero_list)

# Display box graph
st.header("Fuzzy Limits Box Graph")
fig = boxGraph(fuzzy_limits)
st.plotly_chart(fig, use_container_width=True)

# Display fuzzy logic results for selected hero
st.header(f"Fuzzy Logic Performance for Hero: {selected_hero}")
hero_data, message = get_performance_for_hero(selected_hero)
st.write(message)

if not hero_data.empty:
    st.dataframe(hero_data)
else:
    st.warning("Please select a valid hero to see the performance results.")

# Optional: Display summary statistics
st.header("Summary Statistics")
st.write("Summary statistics for all players:")
st.dataframe(all_players.describe())

# Instructions
st.sidebar.header("Instructions")
st.sidebar.write("""
1. Select a hero from the dropdown menu to view the fuzzy logic performance results.
2. The box graph shows the distribution of fuzzy limits (min, mean, max) for each variable across roles.
3. The performance table displays the fuzzy labels for relevant features and the inferred performance (bad, decent, good) for each player using the selected hero.
""")
