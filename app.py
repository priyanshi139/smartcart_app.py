import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, Dataset, Reader
from surprise.model_selection import train_test_split
from surprise import accuracy
from collections import defaultdict
import plotly.express as px
from fuzzywuzzy import process


#  Streamlit UI Setup
st.set_page_config(layout="wide")
st.title("🛍️ SmartCart - Product Recommendation System")


#  Load and Prepare Data 
@st.cache_data
def load_data():
    df = pd.read_excel(r"C:\Users\Priya\Downloads\cleaned_online_retail.xlsx")
    df.dropna(subset=['CustomerID'], inplace=True)
    df['CustomerID'] = df['CustomerID'].astype(str)
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], errors='coerce')
    return df

with st.spinner("Loading data..."):
    df = load_data()

# Product mapping
product_names = df[['StockCode', 'Description']].drop_duplicates().set_index('StockCode')['Description'].to_dict()


#  Precompute Matrices 
@st.cache_data
def prepare_matrices(df):
    interaction_matrix = df.pivot_table(
        index='CustomerID',
        columns='StockCode',
        values='Quantity',
        aggfunc='sum',
        fill_value=0
    )
    customer_ids = interaction_matrix.index.tolist()
    product_codes = interaction_matrix.columns.tolist()
    user_sim_df = pd.DataFrame(
        cosine_similarity(interaction_matrix),
        index=interaction_matrix.index,
        columns=interaction_matrix.index
    )
    item_sim_df = pd.DataFrame(
        cosine_similarity(interaction_matrix.T),
        index=interaction_matrix.columns,
        columns=interaction_matrix.columns
    )
    return interaction_matrix, user_sim_df, item_sim_df, customer_ids, product_codes

with st.spinner("Preparing similarity matrices..."):
    interaction_matrix, user_sim_df, item_sim_df, customer_ids, product_codes = prepare_matrices(df)


#  SVD Model Training 
@st.cache_resource
def train_svd(df):
    reader = Reader(rating_scale=(0.1, df['Quantity'].max()))
    data = Dataset.load_from_df(df[['CustomerID', 'StockCode', 'Quantity']], reader)
    trainset, testset = train_test_split(data, test_size=0.2)
    svd_model = SVD()
    svd_model.fit(trainset)
    predictions = svd_model.test(testset)
    rmse = accuracy.rmse(predictions)
    return svd_model, predictions, rmse

with st.spinner("Training SVD model..."):
    svd_model, predictions, rmse = train_svd(df)

def get_top_n(predictions, n=5):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

svd_top_n = get_top_n(predictions)

#  Recommendation Functions
def recommend_items(user_id, top_n=5):
    if user_id not in interaction_matrix.index:
        return "User not found."
    similar_users = user_sim_df[user_id].sort_values(ascending=False)[1:]
    weighted_sum = np.dot(similar_users, interaction_matrix.loc[similar_users.index])
    scores = pd.Series(weighted_sum, index=interaction_matrix.columns)
    already_purchased = interaction_matrix.loc[user_id]
    scores = scores[already_purchased == 0]
    return scores.sort_values(ascending=False).head(top_n)

def recommend_similar_items(item_code, top_n=5):
    if item_code not in item_sim_df:
        return "Item not found."
    similar_items = item_sim_df[item_code].sort_values(ascending=False)[1:top_n+1]
    return [(code, round(score, 2)) for code, score in similar_items.items()]


#  Streamlit Tabs
tabs = st.tabs(["User-Based", "Item-Based", "SVD-Based", "Dashboard"])

#  User-Based
with tabs[0]:
    st.subheader(" User-Based Collaborative Filtering")
    user_id = st.selectbox("Select Customer ID", customer_ids)
    if st.button("Recommend Products"):
        with st.spinner("Generating recommendations..."):
            result = recommend_items(user_id)
            if isinstance(result, str):
                st.warning(result)
            else:
                for code, score in result.items():
                    st.markdown(f"**{product_names.get(code, 'Unknown Product')}** — Score: `{round(score, 2)}`")

#  Item-Based
with tabs[1]:
    st.subheader(" Item-Based Collaborative Filtering")
    item_code = st.selectbox("Select Product Code", product_codes)
    search = st.text_input("Search Product Name (Optional)")
    if search:
        matches = process.extract(search, list(product_names.values()), limit=3)
        st.write("Top Matches:", matches)
    if st.button("Find Similar Products"):
        with st.spinner("Finding similar items..."):
            result = recommend_similar_items(item_code)
            if isinstance(result, str):
                st.warning(result)
            else:
                for code, score in result:
                    st.markdown(f"**{product_names.get(code, 'Unknown Product')}** — Similarity Score: `{score}`")

#  SVD-Based
with tabs[2]:
    st.subheader(" SVD-Based Matrix Factorization")
    input_user = st.selectbox("Select Customer ID for SVD", customer_ids)
    if st.button("SVD Recommend"):
        with st.spinner("Predicting using SVD..."):
            if input_user in svd_top_n:
                recs = svd_top_n[input_user]
                for item, rating in recs:
                    st.markdown(f"**{product_names.get(item, 'Unknown Product')}** — Estimated Rating: `{round(rating,2)}`")
            else:
                st.warning("User not found in the dataset.")
    st.markdown(f"**RMSE of the model:** `{round(rmse,4)}`")

#  Dashboard
with tabs[3]:
    st.subheader(" Dashboard - Business Insights")
    col1, col2 = st.columns(2)

    with col1:
        top_products = df.groupby('Description')['Quantity'].sum().sort_values(ascending=False).head(10)
        fig = px.bar(top_products, x=top_products.index, y=top_products.values, title="Top 10 Products")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        country_dist = df['Country'].value_counts().head(10)
        fig2 = px.pie(values=country_dist.values, names=country_dist.index, title="Top 10 Countries")
        st.plotly_chart(fig2, use_container_width=True)

    monthly_trend = df.groupby(df['InvoiceDate'].dt.to_period('M'))['Quantity'].sum()
    monthly_trend.index = monthly_trend.index.astype(str)
    fig3 = px.line(monthly_trend, x=monthly_trend.index, y=monthly_trend.values, title="Monthly Purchase Trend")
    st.plotly_chart(fig3, use_container_width=True)
