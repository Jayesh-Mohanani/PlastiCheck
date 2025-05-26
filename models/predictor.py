# Import all required libraries
# Data handling
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px

# Time series & forecasting
import statsmodels.api as sm
from prophet import Prophet

# Machine learning
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.decomposition import PCA
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Synthetic data augmentation
from imblearn.over_sampling import SMOTE
from ctgan import CTGAN

# Web framework
import streamlit as st
# or, if using Flask:
from flask import Flask, render_template, request

# Utility
import os
import warnings
warnings.filterwarnings('ignore')

# Display settings
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
sns.set_style('whitegrid')
import datetime

# Load your processed data and trained model ONCE
df = pd.read_csv(r'C:\Users\jayes\OneDrive\Desktop\College\MSc\CHRIST Data Science\Summer Research Project Work\PlastiCheck\Data\processed_microplastics.csv')
df_log = pd.read_csv(r'C:\Users\jayes\OneDrive\Desktop\College\MSc\CHRIST Data Science\Summer Research Project Work\PlastiCheck\Data\processed_microplastics.csv')

df.columns = df_log.columns.str.strip().str.lower()
df.columns = (df_log.columns.str.strip().str.replace(' ', '_').str.lower())

df_log.columns = df_log.columns.str.strip().str.lower()
df_log.columns = (df_log.columns.str.strip().str.replace(' ', '_').str.lower())

numeric_cols = df.select_dtypes(include='number').columns.drop('year', 'total_ug_per_kg')

corr = df[numeric_cols].corr()

log_cols = numeric_cols
for col in log_cols:
    df_log[col] = np.log1p(df_log[col])

X = df_log[log_cols]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=len(log_cols))
X_pca = pca.fit_transform(X_scaled)

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2, random_state=42)
cluster_labels = kmeans.fit_predict(X_pca[:, :2])

from matplotlib.colors import ListedColormap

PC1 = X_pca[:, 0]
PC2 = X_pca[:, 1]
PC3 = X_pca[:, 2]

kmeans_3d = KMeans(n_clusters=3, random_state=50)
labels_3d = kmeans_3d.fit_predict(X_pca[:, :3])

#3. KMeans Clustering for Regime Detection

#3.1. Apply KMeans with 3 clusters on first 3 PCs
X3 = X_pca[:, :3]
k3 = KMeans(n_clusters=3, random_state=50)
regime_labels_3 = k3.fit_predict(X3)

#3.2. Attach the new regime labels to the DataFrame
df_log['regime_3'] = regime_labels_3

#3.3. Count the number of samples in each cluster
regime_3_counts = df_log['regime_3'].value_counts()

#3.4. Create cluster profiles for each cluster in regime_3
cluster_profiles_3 = (df_log.groupby('regime_3')[numeric_cols].mean().round(2))

#3.5. Compute the overall mean for each cluster
cluster_means = cluster_profiles_3.mean(axis=1)

#3.6. Sort the clusters by overall mean
sorted_labels = cluster_means.sort_values().index

#3.7. Map sorted labels to human-readable regime names
regime_mapping = {
    sorted_labels[0]: 'Low',
    sorted_labels[1]: 'Medium',
    sorted_labels[2]: 'High'
}

#3.8. Apply mapping to your dataframe
df_log['regime'] = df_log['regime_3'].map(regime_mapping)

#3.9. Correlation Analysis
# Compute correlation matrix
corr_matrix = df_log.corr(numeric_only=True)
corr_matrix = corr_matrix.drop('total_ug_per_kg').drop('year', axis=1)

feature_cols = [    
    'cheese', 'yoghurt', 'total_milk', 'fruits', 'refined_grains',
    'whole_grains', 'nuts_and_seeds', 'total_processed_meats',
    'unprocessed_red_meats', 'fish', 'shellfish', 'eggs', 'total_salt',
    'added_sugars', 'non-starchy_vegetables', 'potatoes',
    'other_starchy_vegetables', 'beans_and_legumes'
]

# Prepare the data for modeling
# Select features and target variable
X = df_log[feature_cols]
y = df_log['total_ug_per_kg']

# First, split off the test set (20%)
X_train, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=50)

# Now split the remaining 80% into train (70%) and validation (10%)
# 10% out of 80% is 12.5% of the original data
X_train, X_val, y_train, y_val = train_test_split(X_train, y_temp, test_size=0.125, random_state=50)

xgb = XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=50, max_depth=10, learning_rate=0.1)
xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=True)

y_pred_val_xgb = xgb.predict(X_val)
y_pred_test_xgb = xgb.predict(X_test)

#Save the best model
import joblib

# After fitting your best model (e.g., XGBoost)
joblib.dump(xgb, 'xgboost_microplastic_model.pkl')

# Later, use model.predict(user_input) for predictions:
model = joblib.load('xgboost_microplastic_model.pkl')

xgboost_model = model

def prettify_feature(name):
    return name.replace('_', ' ').title()

def extrapolate_feature(country, feature, start_year, end_year, df):
    sub = df[df['country'] == country][['year', feature]].dropna()
    if sub.shape[0] < 2:
        last_val = sub[feature].iloc[-1]
        return [last_val] * (end_year - start_year + 1)
    coeffs = np.polyfit(sub['year'], sub[feature], 1)
    years = np.arange(start_year, end_year + 1)
    return coeffs[0] * years + coeffs[1]

def run_prediction(user_country, selected_categories, outdir='static/img/tmp'):
    # Ensure output directory exists
    if isinstance(outdir, list):
        outdir = outdir[0]
    os.makedirs(outdir, exist_ok=True)

    # Clean country name
    user_country = user_country.strip()
    df_log['country'] = df_log['country'].str.strip()

    # Prepare country data
    country_df = df_log[df_log['country'] == user_country].sort_values('year')
    train_df = country_df[country_df['year'] < 2018].copy()
    test_df = country_df[country_df['year'] == 2018].copy()

    # Fill missing values for each feature column with median per year
    for feature in feature_cols:
        df_log[feature] = df_log.groupby('year')[feature].transform(lambda x: x.fillna(x.median()))

    # Historical data
    historical_years = [1990, 1995, 2000, 2005, 2010, 2015, 2018]
    historical_df = df_log[(df_log['country'] == user_country) & (df_log['year'].isin(historical_years))]
    historical_years = historical_df['year'].tolist()
    historical_values = historical_df['total_ug_per_kg'].tolist()

    # Forecasting
    forecast_years = list(range(2019, 2031))
    growth_rate = 0.015
    forecast_features = {}
    for feature in feature_cols:
        sub = df_log[(df_log['country'] == user_country) & (~df_log[feature].isna())]
        if sub.empty:
            latest_year = df_log[~df_log[feature].isna()]['year'].max()
            median_val = df_log[df_log['year'] == latest_year][feature].median()
            if np.isnan(median_val):
                median_val = df_log[feature].median()
            base_val = median_val
        else:
            latest_year = sub['year'].max()
            base_val = sub[sub['year'] == latest_year][feature].values[0]
        forecast_features[feature] = [
            base_val * ((1 + growth_rate) ** (year - latest_year)) for year in forecast_years
        ]

    forecast_df = pd.DataFrame({'year': forecast_years})
    for feature in feature_cols:
        forecast_df[feature] = forecast_features[feature]

    forecast_df['Predicted Microplastic (ug/kg)'] = [
        xgboost_model.predict(row[feature_cols].values.reshape(1, -1))[0]
        for _, row in forecast_df.iterrows()
    ]

    all_years = historical_years + forecast_years
    all_values = historical_values + forecast_df['Predicted Microplastic (ug/kg)'].tolist()

    # Highlight year (2025)
    highlight_year = 2025
    highlight_value = forecast_df[forecast_df['year'] == highlight_year]['Predicted Microplastic (ug/kg)'].values[0]

    if not historical_years or not historical_values:
        print(f"No historical data found for {user_country} in the specified years: {historical_years}")
        # Handle this case: skip, use fallback, or raise a more informative error
    else:
        last_hist_year = historical_years[-1]
        last_hist_value = historical_values[-1]

    historical_df = df_log[(df_log['country'] == user_country)]
    if not historical_df.empty:
        last_hist_row = historical_df.sort_values('year').iloc[-1]
        last_hist_year = last_hist_row['year']
        last_hist_value = last_hist_row['total_ug_per_kg']
    else:
        print(f"No historical data found for {user_country}")
        # Handle as needed
        # Since no historical data is found, we can end the program here
        print("Exiting the program.")
        exit()

    # Get the last historical year and value
    last_hist_year = historical_years[-1]
    last_hist_value = historical_values[-1]

    # Add 2018 to the start of forecast line
    forecast_years_with_2018 = [last_hist_year] + forecast_years
    forecast_values_with_2018 = [last_hist_value] + forecast_df['Predicted Microplastic (ug/kg)'].tolist()

    # Plot 1: Forecast
    suffix = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    forecast_path = f'forecast_plot_{suffix}.png'
    forecast_full_path = os.path.join(outdir, forecast_path)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(
        x=historical_years,
        y=historical_values,
        mode='markers+lines',
        name='Historical',
        marker=dict(color='blue', size=8),
        line=dict(dash='dot', color='blue')
    ))
    forecast_years_with_2018 = [historical_years[-1]] + forecast_years
    forecast_values_with_2018 = [historical_values[-1]] + forecast_df['Predicted Microplastic (ug/kg)'].tolist()
    fig_forecast.add_trace(go.Scatter(
        x=forecast_years_with_2018,
        y=forecast_values_with_2018,
        mode='lines',
        name='Forecast (2019–2030)',
        line=dict(color='orange', width=3)
    ))
    fig_forecast.add_trace(go.Scatter(
        x=[highlight_year],
        y=[highlight_value],
        mode='markers+text',
        name='Current (2025)',
        marker=dict(color='red', size=14, symbol='circle'),
        text=[f"{highlight_value:.2f}"],
        textposition="top center",
        showlegend=True
    ))
    fig_forecast.update_layout(
        title=f"Microplastic Forecast for {user_country} (1990–2030)",
        xaxis_title="Year",
        yaxis_title="Predicted Microplastic (ug/kg)",
        xaxis=dict(dtick=5, range=[1988, 2032]),
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    pio.write_image(fig_forecast, forecast_full_path)

    # 9.8 Percentage increase in microplastic consumption for the next five years
    current_value_2025 = highlight_value
    avg_2026_2030 = forecast_df[(forecast_df['year'] >= 2026) & (forecast_df['year'] <= 2030)]['Predicted Microplastic (ug/kg)'].mean()
    percent_increase = ((avg_2026_2030 - current_value_2025) / current_value_2025) * 100

    print(f"\nPredicted microplastic consumption in {user_country} for 2025: {current_value_2025:.2f} ug/kg")
    print(f"Average predicted microplastic consumption in {user_country} for 2026–2030: {avg_2026_2030:.2f} ug/kg")
    print(f"Potential percentage increase in the next five years: {percent_increase:.2f}%")

    # Plot 2: Selected Feature Importance
    importances = xgboost_model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    })
    feature_importance_df['Pretty Feature'] = feature_importance_df['Feature'].apply(prettify_feature)
    feature_importance_df['Weighted Importance'] = [
        imp * 1.25 if feat in selected_categories else imp
        for feat, imp in zip(feature_importance_df['Feature'], feature_importance_df['Importance'])
    ]

    df_selected = feature_importance_df[feature_importance_df['Feature'].isin(selected_categories)].copy()
    df_selected = df_selected.sort_values('Importance', ascending=False)
    selected_path = f'selected_plot_{suffix}.png'
    selected_full_path = os.path.join(outdir, selected_path)
    fig_selected = px.bar(
        df_selected,
        x='Importance',
        y='Pretty Feature',
        orientation='h',
        text_auto='.4f',
        color='Pretty Feature',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title='Contributing Factors to Microplastic Consumption Based on Your Choice',
        labels={'Pretty Feature': 'Food Category', 'Importance': 'Raw Importance (ug/kg)'}
    )
    fig_selected.update_traces(textposition='outside', cliponaxis=False)
    fig_selected.update_layout(
        yaxis=dict(title=dict(text='<b>Food Categories</b>', font=dict(size=16, color='black')), title_standoff=20, automargin=True),
        xaxis=dict(title=dict(text='<b>Raw Importance (ug/kg)</b>', font=dict(size=16, color='black')), title_standoff=20),
        margin=dict(l=200),
        height=600, width=900,
        showlegend=False,
        title=dict(text='<b>Contributing Factors to Microplastic Consumption Based on Your Choice</b>', font=dict(size=22, color='black'), x=0.5, xanchor='center')
    )
    pio.write_image(fig_selected, selected_full_path)

    # Plot 3: All Feature Importance (Weighted)
    feature_importance_df = feature_importance_df.sort_values('Weighted Importance', ascending=False)
    imp_path = f'imp_plot_{suffix}.png'
    imp_full_path = os.path.join(outdir, imp_path)
    fig_importance = px.bar(
        feature_importance_df,
        x='Weighted Importance',
        y='Pretty Feature',
        orientation='h',
        text_auto='.4f',
        color='Pretty Feature',
        color_discrete_sequence=px.colors.qualitative.Plotly,
        title='Possible Contributing Factors to Microplastic Exposure in Various Food Items',
        labels={'Pretty Feature': 'Food Category', 'Weighted Importance': 'Weighted Importance ug/kg'}
    )
    fig_importance.update_traces(textposition='outside', cliponaxis=False)
    fig_importance.update_layout(
        yaxis=dict(title=dict(text='<b>Food Categories</b>', font=dict(size=16, color='black')), title_standoff=20, automargin=True),
        xaxis=dict(title=dict(text='<b>Weighted Importance ug/kg</b>', font=dict(size=16, color='black')), title_standoff=20),
        margin=dict(l=200),
        height=600, width=900,
        showlegend=False,
        title=dict(text='<b>Contributing Factors to Microplastic Exposure in Various Food Items</b>', font=dict(size=22, color='black'), x=0.5, xanchor='center')
    )
    pio.write_image(fig_importance, imp_full_path)


    # Plot 4: Food Category Pie Chart
    category_map = {
        'cheese':                   'Milk and Dairy Products',
        'yoghurt':                  'Milk and Dairy Products',
        'total_milk':               'Milk and Dairy Products',
        'fish':                     'Fish and Seafood',
        'shellfish':                'Fish and Seafood',
        'total_processed_meats':    'Meat and Eggs',
        'unprocessed_red_meats':    'Meat and Eggs',
        'eggs':                     'Meat and Eggs',
        'fruits':                   'Fruits and Vegetables',
        'non_starchy_vegetables':   'Fruits and Vegetables',
        'potatoes':                 'Fruits and Vegetables',
        'other_starchy_vegetables': 'Fruits and Vegetables',
        'refined_grains':           'Nuts and Grains',
        'whole_grains':             'Nuts and Grains',
        'nuts_and_seeds':           'Nuts and Grains',
        'beans_and_legumes':        'Nuts and Grains',
        'total_salt':               'Condiments',
        'added_sugars':             'Condiments'
    }

    feature_importance_df['Category'] = (
        feature_importance_df['Feature']
        .map(category_map)
        .fillna('Other')
    )

    category_importance = (
        feature_importance_df
        .groupby('Category', as_index=False)['Weighted Importance']
        .sum()
        .sort_values('Weighted Importance', ascending=False)
    )

    category_colors = ['blue', 'red', 'green', 'purple', 'orange', 'magenta', 'brown']


    fig_pie = px.pie(
        category_importance,
        names='Category',
        values='Weighted Importance',
        color='Category',
        color_discrete_sequence=category_colors,
        title='Distribution of Major Food Categories in Microplastic Consumption'
    )

    fig_pie.update_traces(
        textinfo='percent',
        textposition='inside',
        textfont=dict(color='white', size=15),
        pull=[0]*len(category_importance)
    )

    fig_pie.update_layout(
        showlegend=True,
        legend_title_text='Major Food Categories',
        legend=dict(
            orientation='v',
            y=1, x=0, xanchor='left', yanchor='top',
            font=dict(size=12),
            title_font=dict(size=13)
        ),
        title=dict(
            text='<b>Distribution of Major Food Categories in Microplastic Consumption</b>',
            font=dict(size=16),
            x=0.5, xanchor='center'
        ),
        margin=dict(t=80, b=40, l=40, r=200),
        width=800,
        height=800
    )

    pie_filename      = f'pie_{suffix}.png'
    pie_full_filepath = os.path.join(outdir, pie_filename)
    pio.write_image(fig_pie, pie_full_filepath)


    # 9.11 Results and Regime Classification
    # 9.11.1 Get the regime for the most recent available year for the selected country
    latest_regime_row = df_log[df_log['country'] == user_country].sort_values('year').iloc[-1]
    regime = latest_regime_row['regime']

    # 9.11.2 Set color and label for display
    regime_display = {
        'Low':    {'color': 'green',  'label': 'Low Microplastic Consumption'},
        'Medium': {'color': 'orange', 'label': 'Medium Microplastic Consumption'},
        'High':   {'color': 'red',    'label': 'High Microplastic Consumption'}
    }
    label_color = regime_display[regime]['color']
    label = regime_display[regime]['label']

    # 9.11.3 Alert if predicted value is close to regime boundary
    regime_bounds = (df_log.groupby('regime')['total_ug_per_kg'].agg(['min', 'max']).sort_values('min').reset_index())

    # Fix overlapping if any: set each lower bound just above the previous upper bound
    adjusted_ranges = []

    prev_max = None

    for idx, row in regime_bounds.iterrows():
        regime = row['regime']
        min_val = row['min']
        max_val = row['max']
    
    # For all but the first regime, set min to previous max + small value to show discontinuity
    if prev_max is not None:
        min_val = round(prev_max + 0.01, 2)
    adjusted_ranges.append((regime, min_val, max_val))
    prev_max = max_val

    # 9.11.4 Find the current regime's index
    regime_idx = regime_bounds[regime_bounds['regime'] == regime].index[0]
    alert_msg = ""
    alert_margin = 0.5  # ug/kg - a Threshold value for alerting

    if regime == 'Low':
        upper_bound = regime_bounds.iloc[regime_idx]['max']
        if abs(current_value_2025 - upper_bound) < alert_margin:
            alert_msg = "Approaching Medium Regime!"
    elif regime == 'Medium':
        lower_bound = regime_bounds.iloc[regime_idx]['min']
        upper_bound = regime_bounds.iloc[regime_idx]['max']
        if abs(current_value_2025 - lower_bound) < alert_margin:
            alert_msg = "Close to Low Regime!"
        elif abs(current_value_2025 - upper_bound) < alert_margin:
            alert_msg = "Approaching High Regime!"
    elif regime == 'High':
        lower_bound = regime_bounds.iloc[regime_idx]['min']
        if abs(current_value_2025 - lower_bound) < alert_margin:
            alert_msg = "Just crossed into High Regime!"
    
    # --- Regime & Alert Logic ---
    # manual_ranges as defined earlier in your code
    manual_ranges = [
        ('Low',    6.00, 7.05,   'green'),
        ('Medium', 7.06, 7.74,   'orange'),
        ('High',   7.75, 8.19,   'red')
    ]

    # Assign regime and label color based on current_value_2025
    label = None
    label_color = None
    for regime_name, min_val, max_val, color in manual_ranges:
        if min_val <= current_value_2025 <= max_val:
            label = regime_name
            label_color = color
            break

    if label is None:
        label = "Unknown"
        label_color = "grey"

    # Find the tuple for the current regime
    regime_tuple = next((r for r in manual_ranges if r[0] == label), None)
    alert_msg = ""
    alert_margin = 0.5  # ug/kg

    if regime_tuple is not None:
        _, min_val, max_val, _ = regime_tuple
        if label == 'Low':
            upper_bound = max_val
            if abs(current_value_2025 - upper_bound) < alert_margin:
                alert_msg = "Approaching Medium Regime!"
            elif label == 'Medium':
                lower_bound = min_val
                upper_bound = max_val
                if abs(current_value_2025 - lower_bound) < alert_margin:
                    alert_msg = "Close to Low Regime!"
                elif abs(current_value_2025 - upper_bound) < alert_margin:
                    alert_msg = "Approaching High Regime!"
            elif label == 'High':
                lower_bound = min_val
                if abs(current_value_2025 - lower_bound) < alert_margin:
                    alert_msg = "Just crossed into High Regime!"
    else:
        alert_msg = "Regime not found!"

    # Build formatted HTML output
    consumption_text = f"Calculated Microplastic Consumption in {user_country} for the year {highlight_year} is: <b>{current_value_2025:.2f} µg/kg</b>"
    regime_html = (
        f'<div style="margin-bottom:8px; font-size: 1.2em">{consumption_text}</div>'
        f'<span style="font-weight:bold; font-size:1.6em">This falls under the </span>'
        f'<span style="color:{label_color}; font-weight:bold; font-size:1.6em">{label} Microplastic Consumption</span>'
        f'<span style="font-weight:bold; font-size:1.6em"> regime.</span>'
    )
    
    if alert_msg:
        regime_html += f' <span style="color:white; background:{label_color}; padding:2px 8px; border-radius:5px; font-weight:bold;">({alert_msg})</span>'

    # --- Return everything needed for the web ---

    user_country = user_country
    current_value_2025 = float(current_value_2025)
    label = label
    alert_msg = alert_msg

    return {
        'user_country': user_country,
        'current_value_2025': current_value_2025,
        'fig1': fig_forecast,
        'fig2': fig_selected,
        'fig3': fig_importance,
        'fig4': fig_pie,
        'regime': label,
        'regime_html': regime_html,
        'percentages': {'increase': round(percent_increase, 2)},
    }