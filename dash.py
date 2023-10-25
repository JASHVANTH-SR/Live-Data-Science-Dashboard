import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the sample dataset
df = pd.read_csv("bank.csv")

st.set_page_config(
    page_title='Real-Time Data Science Dashboard',
    page_icon='✅',
    layout='wide'
)

# Dashboard title
st.title("Real-Time / Live Data Science Dashboard")

# Top-level filters
job_filter = st.selectbox("Select the Job", pd.unique(df['job']))

# Creating a single-element container
placeholder = st.empty()

# Dataframe filter
df = df[df['job'] == job_filter]

# Real-time / live feed simulation
for seconds in range(200):
    # Simulate real-time data updates
    df['age_new'] = df['age'] * np.random.choice(range(1, 5))
    df['balance_new'] = df['balance'] * np.random.choice(range(1, 5))

    # Creating KPIs
    avg_age = np.mean(df['age_new'])
    count_married = int(df[(df["marital"] == 'married')]['marital'].count() + np.random.choice(range(1, 30)))
    balance = np.mean(df['balance_new'])

    with placeholder.container():
        # Create three columns for KPIs
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric(label="Age ⏳", value=round(avg_age), delta=round(avg_age) - 10)
        kpi2.metric(label="Married Count 💍", value=int(count_married), delta=-10 + count_married)
        kpi3.metric(label="A/C Balance ＄", value=f"$ {round(balance, 2)} ", delta=-round(balance / count_married) * 100)

        # Create two columns for charts
        fig_col1, fig_col2 = st.columns(2)
        with fig_col1:
            st.markdown("### Age Distribution")
            fig = px.histogram(data_frame=df, x='age_new', nbins=20)
            st.plotly_chart(fig)

        with fig_col2:
            st.markdown("### Balance Distribution")
            fig2 = px.histogram(data_frame=df, x='balance_new', nbins=20)
            st.plotly_chart(fig2)

        st.markdown("### Detailed Data View")
        st.dataframe(df)

    time.sleep(1)

# Machine Learning Example
st.header("Machine Learning Example")

# Split the data into features and target
X = df[['age', 'balance']]
y = (df['deposit'] == 'yes').astype(int)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy and display confusion matrix
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

st.write(f"Accuracy: {accuracy:.2f}")
st.write("Confusion Matrix:")
st.write(conf_matrix)
