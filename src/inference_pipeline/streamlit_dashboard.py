import streamlit as st
import pandas as pd
from pymongo import MongoClient
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

st.set_page_config(page_title="Review Analytics", layout="wide")

if 'auto_refresh' not in st.session_state:
    st.session_state.auto_refresh = True

col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    st.title("Review Analytics Dashboard")
with col2:
    if st.button("Refresh", key="top_refresh"):
        st.cache_data.clear()
        st.rerun()
with col3:
    auto_refresh = st.toggle("Auto Refresh", value=st.session_state.auto_refresh)
    st.session_state.auto_refresh = auto_refresh

if st.session_state.auto_refresh:
    st.markdown("**Auto-refresh enabled (30s intervals)**")
    st.markdown("""
    <script>
    setTimeout(function(){
        window.location.reload(1);
    }, 30000);
    </script>
    """, unsafe_allow_html=True)

@st.cache_data(ttl=30)
def load_anomalies():
    client = MongoClient('mongodb://mongodb:27017/')
    db = client['review_analytics']
    anomalies = list(db.anomalies.find().sort([("timestamp", -1)]).limit(10))
    client.close()
    return pd.DataFrame(anomalies) if anomalies else pd.DataFrame()

anomalies_df = load_anomalies()
if not anomalies_df.empty:
    if 'dismissed_anomalies' not in st.session_state:
        st.session_state.dismissed_anomalies = set()
    
    for idx, alert in anomalies_df.head(3).iterrows():
        alert_id = str(alert.get('_id', idx))
        
        if alert_id in st.session_state.dismissed_anomalies:
            continue
            
        alert_msg = alert.get('message', 'Anomaly detected')
        alert_severity = alert.get('severity', '')
        
        if alert_severity and isinstance(alert_severity, str):
            severity_str = alert_severity.lower()
        else:
            severity_str = str(alert_severity).lower() if alert_severity else ''
        
        col1, col2 = st.columns([10, 1])
        
        with col1:
            if 'high' in severity_str:
                st.error(f"**ANOMALY DETECTED**: {alert_msg}")
            else:
                st.warning(f"**ANOMALY DETECTED**: {alert_msg}")
        
        with col2:
            if st.button("X", key=f"dismiss_{alert_id}"):
                st.session_state.dismissed_anomalies.add(alert_id)
                st.rerun()

@st.cache_data(ttl=30)
def load_reviews(limit=5000):
    client = MongoClient('mongodb://mongodb:27017/')
    db = client['review_analytics']
    reviews = list(db.reviews.find().sort([("_id", -1)]).limit(limit))
    if reviews:
        df = pd.DataFrame(reviews)
        
        enhanced_data = []
        for review in reviews:
            topic_info = review.get('topic', {})
            if isinstance(topic_info, dict):
                topic_data = {
                    'topic_id': topic_info.get('id', -1),
                    'topic_name': topic_info.get('name', 'Unknown')
                }
            else:
                topic_id = topic_info if topic_info is not None else -1
                topic_data = {
                    'topic_id': topic_id,
                    'topic_name': f'Topic {topic_id}' if topic_id >= 0 else 'Unknown'
                }
            
            cluster_info = review.get('cluster', {})
            if isinstance(cluster_info, dict):
                cluster_data = {
                    'cluster_id': cluster_info.get('id', -1),
                    'cluster_name': cluster_info.get('name', 'Unknown')
                }
            else:
                cluster_id = cluster_info if cluster_info is not None else -1
                cluster_data = {
                    'cluster_id': cluster_id,
                    'cluster_name': f'Cluster {cluster_id}' if cluster_id >= 0 else 'Unknown'
                }
            
            enhanced_data.append({**topic_data, **cluster_data})
        
        enhanced_df = pd.DataFrame(enhanced_data)
        df = pd.concat([df.reset_index(drop=True), enhanced_df], axis=1)
        
        required_columns = ['sentiment', 'cluster', 'review_text', 'product', 'category', 'rating']
        for col in required_columns:
            if col not in df.columns:
                df[col] = -1 if col in ['sentiment', 'cluster'] else 'Unknown'
        
        if 'rating' in df.columns:
            df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
            df['rating'] = df['rating'].clip(1, 5)
            df['rating'] = df['rating'].fillna(3)
        
        df = df[(df['review_text'] != 'Unknown') & 
                (df['review_text'].notna()) & 
                (df['review_text'].str.len() > 0)]
        
        client.close()
        return df
    client.close()
    return pd.DataFrame()



@st.cache_data(ttl=30)
def load_model_evaluations():
    client = MongoClient('mongodb://mongodb:27017/')
    db = client['review_analytics']
    evaluations = list(db.model_evaluations.find().sort([("training_date", -1)]).limit(10))
    client.close()
    return evaluations

df = load_reviews()

if df.empty:
    st.info("No data available. Start the streaming pipeline.")
    st.stop()

st.success(f"Loaded {len(df)} reviews (showing recent data)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    total = len(df)
    st.metric("Total Reviews", total)

with col2:
    if 'sentiment' in df.columns:
        known_sentiment = df[df['sentiment'].isin([0, 1])]
        positive = len(df[df['sentiment'] == 1])
        pos_pct = (positive / len(known_sentiment) * 100) if len(known_sentiment) > 0 else 0
        st.metric("Positive Reviews", f"{positive} ({pos_pct:.1f}%)")

with col3:
    if 'sentiment' in df.columns:
        known_sentiment = df[df['sentiment'].isin([0, 1])]
        negative = len(df[df['sentiment'] == 0])
        neg_pct = (negative / len(known_sentiment) * 100) if len(known_sentiment) > 0 else 0
        st.metric("Negative Reviews", f"{negative} ({neg_pct:.1f}%)")

with col4:
    if 'cluster_id' in df.columns:
        valid_clusters = df[df['cluster_id'] >= 0]['cluster_id'].nunique()
        st.metric("Customer Segments", valid_clusters)
    elif 'cluster' in df.columns:
        if df['cluster'].dtype == 'object':
            valid_clusters = df[df['cluster_id'] >= 0]['cluster_id'].nunique() if 'cluster_id' in df.columns else 0
        else:
            valid_clusters = df[df['cluster'] >= 0]['cluster'].nunique()
        st.metric("Customer Segments", valid_clusters)

st.divider()


st.header("Sentiment Analysis")

col1, col2 = st.columns(2)

with col1:
    if 'sentiment' in df.columns:
        sentiment_counts = df['sentiment'].value_counts()
        

        labels = []
        values = []
        colors = []
        
        if 1 in sentiment_counts.index:
            labels.append('Positive')
            values.append(sentiment_counts[1])
            colors.append('#00cc96')
        
        if 0 in sentiment_counts.index:
            labels.append('Negative')
            values.append(sentiment_counts[0])
            colors.append('#ef553b')
        
        if -1 in sentiment_counts.index:
            labels.append('Unknown')
            values.append(sentiment_counts[-1])
            colors.append('#ababab')
        
        if values:
            fig = px.pie(
                values=values,
                names=labels,
                title="Sentiment Distribution",
                color_discrete_sequence=colors
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No sentiment data available")

with col2:
    if 'topic_name' in df.columns:
        topic_counts = df['topic_name'].value_counts().head(5)
        
        fig = px.bar(
            x=topic_counts.values,
            y=topic_counts.index,
            orientation='h',
            title="Top Topics",
            labels={'x': 'Count', 'y': 'Topic'}
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()


st.header("Topic Analysis")

col1, col2 = st.columns(2)

with col1:
    if 'topic_name' in df.columns:
        topic_data = df[df['topic_id'] >= 0]
        if not topic_data.empty:
            topic_counts = topic_data['topic_name'].value_counts()
            
            fig = px.bar(
                x=topic_counts.index,
                y=topic_counts.values,
                title="Topics Identified",
                labels={'x': 'Topic', 'y': 'Review Count'},
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)

with col2:

    if 'topic_name' in df.columns and 'sentiment' in df.columns:
        topic_data = df[df['topic_id'] >= 0].copy()
        if not topic_data.empty:
            topic_data['sentiment_label'] = topic_data['sentiment'].apply(
                lambda x: 'Negative' if x == 0 else 'Positive' if x == 1 else 'Unknown'
            )
            
            cross_tab = topic_data.groupby(['topic_name', 'sentiment_label']).size().unstack(fill_value=0)
            
            if not cross_tab.empty:
                fig = go.Figure()
                colors = {'Negative': '#ef553b', 'Positive': '#00cc96', 'Unknown': '#ababab'}
                
                for col in cross_tab.columns:
                    fig.add_trace(go.Bar(
                        name=col,
                        x=cross_tab.index,
                        y=cross_tab[col],
                        marker_color=colors.get(col, '#636efa')
                    ))
                
                fig.update_layout(
                    title="Sentiment Distribution by Topic",
                    xaxis_title="Topic",
                    yaxis_title="Count",
                    barmode='group',
                    hovermode='x unified',
                    xaxis_tickangle=45
                )
                st.plotly_chart(fig, use_container_width=True)

st.divider()


st.header("Topic Details")

if 'topic_name' in df.columns and not df[df['topic_id'] >= 0].empty:
    topic_data = df[df['topic_id'] >= 0]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Topic Details")
        topic_summary = topic_data.groupby(['topic_name']).agg({
            'sentiment': ['count', 'mean'],
            'rating': 'mean'
        }).round(2)
        
        topic_summary.columns = ['Review Count', 'Avg Sentiment', 'Avg Rating']
        topic_summary = topic_summary.reset_index().sort_values('Review Count', ascending=False)
        
        st.dataframe(topic_summary, use_container_width=True)
    
    with col2:
        st.subheader("Topic Keywords")
        selected_topic = st.selectbox(
            "Select topic to explore:",
            options=topic_data['topic_name'].unique()
        )
        
        topic_reviews = topic_data[topic_data['topic_name'] == selected_topic]
        
        st.write(f"**{selected_topic}**")
        st.write(f"Total reviews: {len(topic_reviews)}")
        
        sentiment_breakdown = topic_reviews['sentiment'].value_counts()
        sentiment_labels = {0: 'Negative', 1: 'Positive', -1: 'Unknown'}
        
        st.write("Sentiment breakdown:")
        for sentiment, count in sentiment_breakdown.items():
            label = sentiment_labels.get(sentiment, 'Unknown')
            percentage = (count / len(topic_reviews)) * 100
            st.write(f"- {label}: {count} ({percentage:.1f}%)")

st.divider()


st.header("Customer Segmentation")

if 'cluster_id' in df.columns:
    cluster_data = df[df['cluster_id'] >= 0]
else:
    cluster_data = df[df['cluster'].astype(str) != '-1'] if 'cluster' in df.columns else pd.DataFrame()

if not cluster_data.empty:
    col1, col2 = st.columns([1, 2])
    
    with col1:
        if 'cluster_name' in cluster_data.columns:
            cluster_counts = cluster_data['cluster_name'].value_counts()
            
            fig = px.bar(
                x=cluster_counts.index,
                y=cluster_counts.values,
                title="Reviews by Segment",
                labels={'x': 'Customer Segment', 'y': 'Review Count'},
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        elif 'cluster_id' in cluster_data.columns:
            cluster_counts = cluster_data['cluster_id'].value_counts().sort_index()
            
            fig = px.bar(
                x=[f"Segment {int(i)}" for i in cluster_counts.index],
                y=cluster_counts.values,
                title="Reviews by Segment",
                labels={'x': 'Customer Segment', 'y': 'Review Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No cluster data available")
    
    with col2:
        st.subheader("Segment Details")
        
        if 'cluster_name' in cluster_data.columns:
            unique_clusters = sorted(cluster_data['cluster_name'].unique())
            if unique_clusters:
                selected_cluster_name = st.selectbox(
                    "Select segment to review",
                    options=unique_clusters
                )
                
                cluster_reviews = cluster_data[cluster_data['cluster_name'] == selected_cluster_name]
                st.write(f"**{selected_cluster_name}** - {len(cluster_reviews)} reviews")
            else:
                st.info("No segments available")
        elif 'cluster_id' in cluster_data.columns:
            unique_clusters = sorted(cluster_data['cluster_id'].unique())
            if unique_clusters:
                selected_cluster = st.selectbox(
                    "Select segment to review",
                    options=unique_clusters
                )
                
                cluster_reviews = cluster_data[cluster_data['cluster_id'] == selected_cluster]
                st.write(f"**Segment {int(selected_cluster)}** - {len(cluster_reviews)} reviews")
            else:
                st.info("No segments available")
        else:
            st.info("No cluster data available")
else:
    st.info("No customer segmentation data available yet")

st.divider()

st.header("System Status")

col1, col2 = st.columns(2)

with col2:
    st.subheader("Model Performance")
    evaluations = load_model_evaluations()
    
    if evaluations:
        latest = evaluations[0]
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Accuracy", f"{latest.get('accuracy', 0):.1%}")
            st.metric("F1 Score", f"{latest.get('f1_score', 0):.3f}")
            st.metric("Precision", f"{latest.get('precision', 0):.3f}")
        
        with col2:
            st.metric("Recall", f"{latest.get('recall', 0):.3f}")
            if latest.get('topic_modeling_logPerplexity_score'):
                st.metric("Topic logPerplexity Score", f"{latest.get('topic_modeling_logPerplexity_score', 0):.3f}")
            st.metric("Clustering Silhouette Score", f"{latest.get('clustering_Silhouette_score', 0):.3f}")
        
        training_date = latest.get('training_date', 'Unknown')
        if training_date != 'Unknown':
            from datetime import datetime
            try:
                dt = datetime.fromisoformat(training_date.replace('Z', '+00:00'))
                st.caption(f"Last trained: {dt.strftime('%Y-%m-%d %H:%M')}")
            except:
                st.caption(f"Last trained: {training_date[:19]}")
    else:
        st.info("No model evaluation data available")



st.divider()

st.header("Model Training History")

evaluations = load_model_evaluations()

if evaluations:
    eval_data = []
    for eval_result in evaluations:
        row_data = {
            'Date': eval_result.get('training_date', '')[:19],
            'Accuracy': f"{eval_result.get('accuracy', 0):.1%}",
            'F1 Score': f"{eval_result.get('f1_score', 0):.3f}",
            'Precision': f"{eval_result.get('precision', 0):.3f}",
            'Recall': f"{eval_result.get('recall', 0):.3f}",
            'Clustering Silhouette Score': f"{eval_result.get('clustering_Silhouette_score', 0):.3f}"
        }
        
        if eval_result.get('topic_modeling_logPerplexity_score'):
            row_data['Topic logPerplexity Score'] = f"{eval_result.get('topic_modeling_logPerplexity_score', 0):.3f}"
        
        eval_data.append(row_data)
    
    eval_df = pd.DataFrame(eval_data)
    st.dataframe(eval_df, use_container_width=True)
else:
    st.info("No training history available")

st.divider()

st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.caption(f"Displaying {len(df)} reviews (auto-refresh: {'ON' if st.session_state.auto_refresh else 'OFF'})")