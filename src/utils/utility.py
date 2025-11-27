import json
import sys
import os
from pyspark.sql import functions as F
from pyspark.sql.types import ArrayType, DoubleType
from pyspark.ml.feature import VectorAssembler
import numpy as np

def extract_clustering_features(df_topics, review_col="Review", summary_col="Summary", sentiment_col="sentiment_label", rate_col="Rate_num"):
    features_df = df_topics.withColumn(
        "text_len", F.coalesce(F.length(F.col("text_combined")).cast("double"), F.lit(0.0))
    ).withColumn(
        "sentiment_score", F.coalesce(F.col(sentiment_col).cast("double"), F.lit(0.0))
    ).withColumn(
        "review_len", F.coalesce(F.length(F.coalesce(F.col(review_col), F.lit(""))).cast("double"), F.lit(0.0))
    ).withColumn(
        "summary_len", F.coalesce(F.length(F.coalesce(F.col(summary_col), F.lit(""))).cast("double"), F.lit(0.0))
    ).withColumn(
        "rate_score", F.coalesce(F.col(rate_col).cast("double"), F.lit(0.0))
    ).withColumn(
        "word_count", F.coalesce(F.size(F.split(F.coalesce(F.col("text_combined"), F.lit("")), " ")).cast("double"), F.lit(1.0))
    )
        
    def extract_topic_probs(topic_dist):
        if topic_dist is None:
            return [0.2] * 5 
        return topic_dist.toArray().tolist() if hasattr(topic_dist, 'toArray') else [0.2] * 5 
    
    topic_probs_udf = F.udf(extract_topic_probs, ArrayType(DoubleType()))
    features_df = features_df.withColumn("topic_probs", topic_probs_udf(F.col("topicDistribution")))
    
    for i in range(5):
        features_df = features_df.withColumn(f"topic_{i}", 
                                           F.coalesce(F.col("topic_probs")[i], F.lit(0.2)))
    
    features_df = features_df.filter(
        F.col("text_len").isNotNull() & 
        F.col("sentiment_score").isNotNull() &
        F.col("rate_score").isNotNull() &
        (F.col("text_len") > 0)
    )
    
    feature_cols = [
        "text_len", "sentiment_score", "review_len", "summary_len", 
        "rate_score", "word_count"
    ] + [f"topic_{i}" for i in range(5)]
    
    for col in feature_cols:
        features_df = features_df.fillna(0.0, subset=[col])
    
    return features_df, feature_cols


def preprocess_data(df, is_training=True):   
    
    if is_training and "Sentiment" in df.columns:
        df_labeled = df.withColumn(
            "sentiment_label",
            F.when(F.lower(F.col("Sentiment")).contains("negative"), 0)
             .when(F.lower(F.col("Sentiment")).contains("positive"), 1)
             .when(F.lower(F.col("Sentiment")).contains("neutral"), -1)
             .otherwise(-1)
        )
    else:
        df_labeled = df.withColumn("sentiment_label", F.lit(0))
    
    df_combined = clean_and_combine_text(df_labeled, "Review", "Summary")
    
    df_features = df_combined.withColumn("review_length", 
                                         F.length(F.coalesce(F.col("Review"), F.lit(""))).cast("float"))
    df_features = df_features.withColumn("summary_length", 
                                         F.length(F.coalesce(F.col("Summary"), F.lit(""))).cast("float"))
    
    if "Rate" in df.columns:
        df_features = df_features.withColumn("Rate_num", F.col("Rate").cast("float"))
    else:
        df_features = df_features.withColumn("Rate_num", F.lit(0.0))


    if is_training:
        df_clean = df_features.filter(
            (F.col("text_combined").isNotNull()) &
            (F.length(F.col("text_combined")) > 1) &
            (F.col("sentiment_label") >= 0)
        )
    else:
        df_clean = df_features.filter(
            (F.col("text_combined").isNotNull()) &
            (F.length(F.col("text_combined")) > 1)
        )
    
    numeric_cols = ["Rate_num", "review_length", "summary_length"]
    df_clean = df_clean.fillna(0.0, subset=numeric_cols)
    
    row_count = df_clean.count()
    print(f"Preprocessing complete. Rows: {row_count}")
    
    if is_training:
        return df_clean.cache()
    else:
        return df_clean


def clean_and_combine_text(df, review_col="Review", summary_col="Summary"):
    regpat_non_alpha = r"[^a-zA-Z\s]"
    regpat_extra_space = r"\s+"
    regpat_nan = r"\bnan\b"
    
    df_cleaned = df.withColumn("cleaned_review", F.lower(F.coalesce(F.col(review_col), F.lit(""))))
    df_cleaned = df_cleaned.withColumn("cleaned_review", 
                                       F.regexp_replace("cleaned_review", regpat_non_alpha, ""))
    df_cleaned = df_cleaned.withColumn("cleaned_review", 
                                       F.regexp_replace("cleaned_review", regpat_nan, ""))
    df_cleaned = df_cleaned.withColumn("cleaned_review", 
                                       F.regexp_replace("cleaned_review", regpat_extra_space, " "))
    df_cleaned = df_cleaned.withColumn("cleaned_review", F.trim(F.col("cleaned_review")))
    
    df_cleaned = df_cleaned.withColumn("cleaned_summary", F.lower(F.coalesce(F.col(summary_col), F.lit(""))))
    df_cleaned = df_cleaned.withColumn("cleaned_summary", 
                                       F.regexp_replace("cleaned_summary", regpat_non_alpha, ""))
    df_cleaned = df_cleaned.withColumn("cleaned_summary", 
                                       F.regexp_replace("cleaned_summary", regpat_nan, ""))
    df_cleaned = df_cleaned.withColumn("cleaned_summary", 
                                       F.regexp_replace("cleaned_summary", regpat_extra_space, " "))
    df_cleaned = df_cleaned.withColumn("cleaned_summary", F.trim(F.col("cleaned_summary")))
    
    df_combined = df_cleaned.withColumn("text_combined", 
                                        F.concat_ws(" ", "cleaned_review", "cleaned_summary"))
    
    return df_combined