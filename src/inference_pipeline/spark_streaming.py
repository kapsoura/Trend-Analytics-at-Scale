import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml import PipelineModel
from pymongo import MongoClient

from utils.utility import (preprocess_data, extract_clustering_features)
import json
from datetime import datetime


def get_spark():
    return SparkSession.builder \
        .appName("FlipkartReviewStreaming") \
        .config("spark.master", "spark://spark-master:7077") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.memory", "1g") \
        .config("spark.default.parallelism", "6") \
        .config("spark.sql.shuffle.partitions", "6") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .getOrCreate()

CLUSTER_INTERPRETATIONS = {}
TOPIC_INTERPRETATIONS = {}

def load_interpretations_from_hdfs(spark):
    global CLUSTER_INTERPRETATIONS, TOPIC_INTERPRETATIONS
    
    try:
        cluster_lines = spark.read.text("hdfs://namenode:9000/models/cluster_interpretations").collect()
        cluster_json = ''.join([row[0] for row in cluster_lines])
        cluster_data = json.loads(cluster_json)
        CLUSTER_INTERPRETATIONS = {int(k): v['name'] for k, v in cluster_data.items()}
        print(f"Loaded {len(CLUSTER_INTERPRETATIONS)} cluster interpretations from HDFS")
    except Exception as e:
        print(f"Failed to load cluster interpretations: {e}")
        CLUSTER_INTERPRETATIONS = {0: "Critical Issues", 1: "Satisfied Customers", 2: "Mixed Feedback"}
    
    try:
        topic_lines = spark.read.text("hdfs://namenode:9000/models/topic_interpretations").collect()
        topic_json = ''.join([row[0] for row in topic_lines])
        topic_data = json.loads(topic_json)
        TOPIC_INTERPRETATIONS = {int(k): v['name'] for k, v in topic_data.items()}
        print(f"Loaded {len(TOPIC_INTERPRETATIONS)} topic interpretations from HDFS")
    except Exception as e:
        print(f"Failed to load topic interpretations: {e}")
        TOPIC_INTERPRETATIONS = {0: "Product Quality", 1: "Delivery & Shipping", 2: "Price & Value"}

def get_cluster_name(cluster_id):
    return CLUSTER_INTERPRETATIONS.get(cluster_id, f"Cluster {cluster_id}")

def get_topic_name(topic_id):
    return TOPIC_INTERPRETATIONS.get(topic_id, f"Topic {topic_id}")



def detect_anomalies_in_batch(df):
    if df.isEmpty():
        return
    
    total_reviews = df.count()
    negative_reviews = df.filter(F.col("sentiment_pred") == 0).count()
    
    alerts = []
        
    if total_reviews > 0:
        neg_pct = (negative_reviews / total_reviews) * 100
        if neg_pct > 70:
            alerts.append({
                'type': 'high_negative', 
                'message': f'High negative sentiment: {neg_pct:.0f}% in batch',
                'timestamp': datetime.utcnow().isoformat()
            })
    
    if alerts:
        try:
            client = MongoClient('mongodb://mongodb:27017/')
            db = client['review_analytics']
            db.anomalies.insert_many(alerts)
            client.close()
            print(f"Detected {len(alerts)} anomalies in batch")
        except:
            print("Failed to save anomalies")

def load_models():
    models = {}
    model_paths = {
        'sentiment': 'hdfs://namenode:9000/models/sentiment_model',
        'topic': 'hdfs://namenode:9000/models/topic_modeling_model', 
        'clustering': 'hdfs://namenode:9000/models/clustering_model'
    }
    
    for name, path in model_paths.items():
        try:
            models[name] = PipelineModel.load(path)
            print(f"Loaded {name} model")
        except:
            print(f"Could not load {name} model")
    
    return models

def apply_models(df, models):
    if 'sentiment' in models:
        df = models['sentiment'].transform(df)
        df = df.withColumnRenamed("prediction", "sentiment_pred")
        
        df = df.withColumn("sentiment_pred", 
                          F.when(F.col("sentiment_pred").isin([0, 1]), F.col("sentiment_pred"))
                          .otherwise(F.lit(-1)))
        
        df = df.withColumn("sentiment_label", 
                          F.when(F.col("sentiment_pred") == 1, F.lit("Positive"))
                          .when(F.col("sentiment_pred") == 0, F.lit("Negative"))
                          .otherwise(F.lit("Unknown")))
        
        df = df.drop("words", "words_filtered", "bigrams", "raw_features", "text_features", 
                    "numeric_features", "scaled_numeric", "features", "rawPrediction", "probability")
    
    if 'topic' in models:
        df = models['topic'].transform(df)
        from pyspark.sql.functions import udf
        from pyspark.sql.types import IntegerType
        import numpy as np
        
        def get_dominant_topic(topic_dist):
            if topic_dist is None:
                return 0
            return int(np.argmax(topic_dist.toArray()))
        
        get_topic_udf = udf(get_dominant_topic, IntegerType())
        df = df.withColumn("topic_pred", get_topic_udf(F.col("topicDistribution")))
        df = df.drop("words", "filtered", "features")
    else:
        df = df.withColumn("topic_pred", F.lit(0))
    
    if 'clustering' in models:
        if "topicDistribution" not in df.columns:
            from pyspark.ml.linalg import Vectors
            default_topic_dist = Vectors.dense([0.2, 0.2, 0.2, 0.2, 0.2])
            df = df.withColumn("topicDistribution", F.lit(default_topic_dist))
        
        df_for_clustering = df.withColumn("sentiment_label", F.col("sentiment_pred"))
        df_with_features, feature_cols = extract_clustering_features(
            df_for_clustering, 
            review_col="Review", 
            summary_col="Summary", 
            sentiment_col="sentiment_label", 
            rate_col="Rate_num"
        )
        
        df = models['clustering'].transform(df_with_features)
        df = df.withColumnRenamed("prediction", "cluster_pred")
        
        cleanup_cols = ["text_len", "sentiment_score", "review_len", "summary_len", 
                      "rate_score", "word_count", "topic_probs", "raw_features", "clustering_features"] + \
                      [f"topic_{i}" for i in range(5)]
        for col in cleanup_cols:
            if col in df.columns:
                df = df.drop(col)
    else:
        df = df.withColumn("cluster_pred", F.expr("abs(hash(review_id)) % 3"))
    
    return df

def save_results(df):
    def save_partition(partition):
        client = MongoClient('mongodb://mongodb:27017/')
        db = client['review_analytics']
        
        batch = []
        for row in partition:
            try:
                rating_val = float(row.Rate) if row.Rate else 3.0
                rating_val = max(1.0, min(5.0, rating_val))
            except (ValueError, TypeError):
                rating_val = 3.0
            
            record = {
                'review_id': row.review_id,
                'product': row.Product,
                'category': row.Category,
                'review_text': row.Review,
                'summary': row.Summary,
                'rating': rating_val,
                'sentiment': getattr(row, 'sentiment_pred', -1),
                'sentiment_label': getattr(row, 'sentiment_label', 'Unknown'),
                'topic': {
                    'id': getattr(row, 'topic_pred', -1),
                    'name': get_topic_name(getattr(row, 'topic_pred', -1))
                },
                'cluster': {
                    'id': getattr(row, 'cluster_pred', -1),
                    'name': get_cluster_name(getattr(row, 'cluster_pred', -1))
                },
                'processed_at': datetime.utcnow(),
                'timestamp': row.timestamp
            }
            batch.append(record)
        
        if batch:
            try:
                db.reviews.insert_many(batch, ordered=False)
            except Exception as e:
                print(f"Error: {e}")
        
        client.close()
        return iter([len(batch)])
    
    df.rdd.mapPartitions(save_partition).collect()

def process_batch(df, batch_id):
    if df.isEmpty():
        return
    
    df_processed = preprocess_data(df, is_training=False)
    df = apply_models(df_processed, MODELS)
    
    detect_anomalies_in_batch(df)
    save_results(df)
    
    print(f"Batch {batch_id}: {df.count()} reviews")

MODELS = {}

def main():
    global MODELS
    
    spark = get_spark()
    spark.sparkContext.setLogLevel("WARN")
    
    MODELS = load_models()
    load_interpretations_from_hdfs(spark)
    
    schema = StructType([
        StructField("review_id", StringType()),
        StructField("timestamp", StringType()),
        StructField("Product", StringType()),
        StructField("Category", StringType()),
        StructField("Review", StringType()),
        StructField("Summary", StringType()),
        StructField("Rate", StringType()),
        StructField("product_price", StringType())
    ])
    
    df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "broker:9092") \
        .option("subscribe", "flipkart_product_reviews") \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()
    
    df = df.select(F.from_json(F.col("value").cast("string"), schema).alias("data")).select("data.*")
    
    query = df.writeStream \
        .foreachBatch(process_batch) \
        .trigger(processingTime='10 seconds') \
        .outputMode("append") \
        .option("checkpointLocation", "/tmp/kafka-checkpoint") \
        .start()
    
    print("Streaming started")
    query.awaitTermination()

if __name__ == '__main__':
    main()