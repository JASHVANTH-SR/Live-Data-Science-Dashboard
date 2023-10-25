import streamlit as st
import numpy as np
import pandas as pd
import time
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# Read the sample dataset
df = pd.read_csv("bank.csv")

st.set_page_config(
    page_title='Data Science Moving Dashboard',
    page_icon='✅',
    layout='wide'
)

# Dashboard title
st.title("Data Science Moving Dashboard")

# Top-level filters
job_filter = st.selectbox("Select the Job", pd.unique(df['job']))
marital_filter = st.selectbox("Select Marital Status", pd.unique(df['marital']))

# Creating a single-element container
placeholder = st.empty()

# Dataframe filters
df = df[df['job'] == job_filter]
df = df[df['marital'] == marital_filter]

# Real-time / live feed simulation
for seconds in range(200):
    # Simulate real-time data updates
    df['age_new'] = df['age'] + np.random.choice(range(1, 5))
    df['balance_new'] = df['balance'] + np.random.choice(range(1, 5))

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

# Exploratory Data Analysis (EDA)
st.header("Exploratory Data Analysis")

# Pairplot
st.markdown("### Pairplot")
sns.pairplot(df, hue="deposit", diag_kind="kde")
st.pyplot()

# Correlation Heatmap
st.markdown("### Correlation Heatmap")
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True)
st.pyplot()

# Bar Chart for Deposit
st.markdown("### Deposit Distribution")
deposit_counts = df['deposit'].value_counts()
fig_deposit = px.bar(x=deposit_counts.index, y=deposit_counts.values, labels={'x':'Deposit', 'y':'Count'})
st.plotly_chart(fig_deposit)

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
