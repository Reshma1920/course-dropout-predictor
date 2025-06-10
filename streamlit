# 📊 Streamlit App: Predicting Course Dropout

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Course Dropout Predictor", layout="wide")
st.title("🎓 Predicting Online Course Dropout")

st.markdown("Upload your MOOC interaction dataset to analyze learner behavior and predict dropout risk.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df['time'] = pd.to_datetime(df['time'])

    # --- Aggregate User-Level Data ---
    agg_df = df.groupby('enroll_id').agg(
        total_actions=('action', 'count'),
        unique_actions=('action', pd.Series.nunique),
        session_count=('session_id', pd.Series.nunique),
        first_activity=('time', 'min'),
        last_activity=('time', 'max')
    ).reset_index()

    agg_df['activity_span_days'] = (agg_df['last_activity'] - agg_df['first_activity']).dt.days
    agg_df['dropout'] = df.groupby('enroll_id')['truth'].first().values
    agg_df.drop(columns=['first_activity', 'last_activity'], inplace=True)

    st.subheader("📈 Feature Summary")
    st.write(agg_df.describe())

    # --- Model Training ---
    features = ['unique_actions', 'session_count', 'activity_span_days']
    X = agg_df[features]
    y = agg_df['dropout']

    log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
    log_model.fit(X, y)

    threshold = st.slider("Set Dropout Prediction Threshold", 0.0, 1.0, 0.5)
    proba = log_model.predict_proba(X)[:, 1]
    agg_df['dropout_pred'] = (proba >= threshold).astype(int)
    
    st.subheader("🧠 Logistic Regression Prediction")
    st.write(agg_df[['enroll_id', 'dropout', 'dropout_pred']].head())

    # --- Clustering ---
    n_clusters = st.slider("Select number of user segments (clusters)", 2, 6, 3)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    agg_df['cluster'] = kmeans.fit_predict(agg_df[['total_actions', 'session_count', 'activity_span_days']])

    st.subheader("📊 Learner Segments")
    st.write(agg_df.groupby('cluster')[['total_actions', 'session_count', 'activity_span_days', 'dropout']].mean())

    fig, ax = plt.subplots()
    sns.scatterplot(data=agg_df, x='session_count', y='total_actions', hue='cluster', palette='Set2', ax=ax)
    plt.title("Clusters by Session Count and Engagement")
    st.pyplot(fig)

    # --- Decision Tree ---
    tree_model = DecisionTreeClassifier(max_depth=4, random_state=42)
    tree_model.fit(X, y)

    st.subheader("🌳 Decision Tree")
    fig_tree, ax_tree = plt.subplots(figsize=(10, 6))
    plot_tree(tree_model, feature_names=features, class_names=['Retained', 'Dropped'], filled=True, rounded=True, ax=ax_tree)
    st.pyplot(fig_tree)

    # --- Engagement Regression ---
    st.subheader("📉 Session Count vs Engagement")
    from sklearn.linear_model import LinearRegression
    model_simple = LinearRegression()
    model_simple.fit(agg_df[['session_count']], agg_df['total_actions'])
    y_pred_simple = model_simple.predict(agg_df[['session_count']])

    fig_reg, ax_reg = plt.subplots()
    ax_reg.scatter(agg_df['session_count'], agg_df['total_actions'], alpha=0.4)
    ax_reg.plot(agg_df['session_count'], y_pred_simple, color='red')
    ax_reg.set_xlabel("Session Count")
    ax_reg.set_ylabel("Total Actions")
    ax_reg.set_title("Simple Linear Regression")
    st.pyplot(fig_reg)

else:
    st.info("📂 Please upload a CSV file to begin analysis.")
