import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, '..')
sys.path.append(src_dir)

import time
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.feature import (
    Tokenizer, StopWordsRemover, CountVectorizer, 
    RegexTokenizer, VectorAssembler, NGram, IDF, StandardScaler
)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, ClusteringEvaluator
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.clustering import LDA, KMeans
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from utils.utility import (preprocess_data, extract_clustering_features)
from pymongo import MongoClient
import json

def create_spark_session():
    return SparkSession.builder \
        .appName("SentimentAnalysisTraining") \
        .config("spark.master", "spark://spark-master:7077") \
        .config("spark.executor.cores", "2") \
        .config("spark.executor.memory", "2g") \
        .config("spark.driver.memory", "2g") \
        .config("spark.default.parallelism", "6") \
        .config("spark.sql.shuffle.partitions", "6") \
        .config("spark.hadoop.fs.defaultFS", "hdfs://namenode:9000") \
        .config("spark.ui.port", "4060") \
        .getOrCreate()

def save_evaluation_results(acc, f1, precision, recall, cluster_score, topic_score=None):
    client = MongoClient('mongodb://mongodb:27017/')
    db = client['review_analytics']
    
    evaluation_result = {
        'model_type': 'sentiment_analysis',
        'accuracy': round(float(acc), 4),
        'f1_score': round(float(f1), 4),
        'precision': round(float(precision), 4),
        'recall': round(float(recall), 4),
        'clustering_Silhouette_score': round(float(cluster_score), 4),
        'topic_modeling_logPerplexity_score': round(float(topic_score), 4) if topic_score else None,
        'training_date': datetime.now().isoformat(),
        'status': 'completed'
    }
    
    db.model_evaluations.insert_one(evaluation_result)
    client.close()
    print("Evaluation results saved to MongoDB")

def create_hdfs_directories(spark):
    directories = ["/data", "/data/training", "/models", "/models/metadata"]
    
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.defaultFS", "hdfs://namenode:9000")
    fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    
    for directory in directories:
        path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(directory)
        if not fs.exists(path):
            fs.mkdirs(path)

def upload_csv_to_hdfs(local_csv_path, hdfs_csv_path, spark):
    if not os.path.exists(local_csv_path):
        raise FileNotFoundError(f"Local CSV not found: {local_csv_path}")
    
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.defaultFS", "hdfs://namenode:9000")
    fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    
    local_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(local_csv_path)
    hdfs_path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path(hdfs_csv_path)
    
    if not fs.exists(hdfs_path):
        fs.copyFromLocalFile(local_path, hdfs_path)
    else:
        print(f"File already exists in HDFS: {hdfs_csv_path}")

def load_csv_from_hdfs(hdfs_csv_path, spark):
    full_path = f"hdfs://namenode:9000{hdfs_csv_path}"
    df = spark.read.option("header", "true").option("inferSchema", "true").csv(full_path)
    return df

def evaluate_model_detailed(predictions):    
    evaluator_acc = MulticlassClassificationEvaluator(
        labelCol="sentiment_label", predictionCol="prediction", metricName="accuracy"
    )
    accuracy = evaluator_acc.evaluate(predictions)
    
    evaluator_f1 = MulticlassClassificationEvaluator(
        labelCol="sentiment_label", predictionCol="prediction", metricName="f1"
    )
    f1_score = evaluator_f1.evaluate(predictions)
    
    evaluator_precision = MulticlassClassificationEvaluator(
        labelCol="sentiment_label", predictionCol="prediction", metricName="weightedPrecision"
    )
    precision = evaluator_precision.evaluate(predictions)
    
    evaluator_recall = MulticlassClassificationEvaluator(
        labelCol="sentiment_label", predictionCol="prediction", metricName="weightedRecall"
    )
    recall = evaluator_recall.evaluate(predictions)   
    
    return accuracy, f1_score, precision, recall

def train_sentiment_model(spark, processed_df):
    print("Starting sentiment model training...")
    

    train_data, val_data, test_data = processed_df.randomSplit([0.70, 0.15, 0.15], seed=42)
    print(f"Train: {train_data.count()}, Val: {val_data.count()}, Test: {test_data.count()}")
    
    numeric_feature_cols = ["Rate_num", "review_length", "summary_length"]
    
    tokenizer = Tokenizer(inputCol="text_combined", outputCol="words")
    stop_remover = StopWordsRemover(inputCol="words", outputCol="words_filtered")
    ngram = NGram(n=2, inputCol="words_filtered", outputCol="bigrams")
    cv = CountVectorizer(inputCol="bigrams", outputCol="raw_features", vocabSize=50000, minDF=3)
    idf = IDF(inputCol="raw_features", outputCol="text_features")
    numeric_assembler = VectorAssembler(inputCols=numeric_feature_cols, outputCol="numeric_features")
    scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric", withMean=True, withStd=True)
    final_assembler = VectorAssembler(inputCols=["text_features", "scaled_numeric"], outputCol="features")
    lr = LogisticRegression(labelCol="sentiment_label", featuresCol="features", maxIter=50)
    
    pipeline = Pipeline(stages=[tokenizer, stop_remover, ngram, cv, idf, numeric_assembler, scaler, final_assembler, lr])
    
    paramGrid = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]) \
        .build()
    
    crossval = CrossValidator(
        estimator=pipeline,
        estimatorParamMaps=paramGrid,
        evaluator=MulticlassClassificationEvaluator(labelCol="sentiment_label"),
        numFolds=3
    )
    
    print("Training model with cross-validation...")
    model = crossval.fit(train_data)
    
    best_model = model.bestModel
    best_lr_model = best_model.stages[-1]
    
    print(f"Best regParam: {best_lr_model.getRegParam()}")
    print(f"Best elasticNetParam: {best_lr_model.getElasticNetParam()}")
    
    predictions = best_model.transform(val_data)
    accuracy, f1_score, precision, recall = evaluate_model_detailed(predictions)
    
    model_path = save_model_to_hdfs("sentiment", "LogisticRegression", best_model, accuracy, spark)
    print(f"Model saved to HDFS: {model_path}")
    
    return best_model, accuracy, f1_score, precision, recall

def create_topic_pipeline():
    tokenizer = RegexTokenizer(inputCol="text_combined", outputCol="words", pattern="\\W+", minTokenLength=3)
    
    stopwords = StopWordsRemover.loadDefaultStopWords("english")
    sentiment_words = ['good', 'bad', 'best', 'worst', 'excellent', 'poor', 'great', 'terrible', 
                      'awesome', 'horrible', 'amazing', 'awful', 'nice', 'fabulous', 'perfect',
                      'wonderful', 'mindblowing', 'terrific', 'super', 'happy', 'love', 'hate']
    
    
    
    problematic_words_base = ['flipkart', 'amazon', 'really', 'brilliant', 'expectations', 'disappointed',
                             'wow', 'thank', 'thanks', 'one', 'also', 'every', 'simply', 'must',
                             'working', 'used', 'use', 'using']
    
    
    all_stopwords = set(stopwords)  
    
    
    for word in sentiment_words:
        all_stopwords.update([word.lower(), word.upper(), word.capitalize()])
    
    
    for word in problematic_words_base:
        all_stopwords.update([word.lower(), word.upper(), word.capitalize()])
    
    
    final_stopwords = [word for word in all_stopwords if len(word) >= 2]
    
    remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=final_stopwords)
    cv = CountVectorizer(inputCol="filtered", outputCol="features", vocabSize=5000, minDF=5, maxDF=0.8)
    lda = LDA(featuresCol="features", k=5, maxIter=20, optimizer="online")
    return Pipeline(stages=[tokenizer, remover, cv, lda])

def interpret_topic_meaning(top_words):    
    words = [word for word, _ in top_words[:10]]
    word_set = set(words)
    
    topic_patterns = {
        'Delivery & Shipping': {'delivery', 'shipping', 'arrived', 'package', 'delayed', 'received', 'courier'},
        'Price & Value': {'purchase', 'price', 'expensive', 'cheap', 'cost', 'money', 'affordable', 'budget', 'deal', 'rupees', 'rs', 'value', 'money', 'penny'},
        'Customer Service': {'service', 'support', 'help', 'staff', 'call', 'response', 'representative', 'care'},
        'Product Features': {'battery', 'camera', 'screen', 'memory', 'color', 'design', 'feature', 'specification', 'display', 'sound', 'machine'},
        'Product Quality': {'quality', 'defective', 'durable', 'broken', 'damaged', 'condition', 'build', 'product'},
        'General Sentiment': {'excellent', 'good', 'bad', 'nice', 'best', 'worst', 'great', 'awesome', 'terrific', 'fabulous', 'super', 'poor', 'perfect'}
    }
    
    best_match = None
    max_overlap = 0
    
    for topic_name, keywords in topic_patterns.items():
        overlap = len(word_set.intersection(keywords))
        if overlap > max_overlap:
            max_overlap = overlap
            best_match = topic_name
    
    if max_overlap < 2:
        best_match = f"{words[0].title()} & {words[1].title()}"
    
    return best_match

def extract_topic_interpretations(lda_model, cv_model, spark, num_topics=5, num_words=10):
    vocab = cv_model.vocabulary
    topics_matrix = lda_model.describeTopics(num_words)
    topic_interpretations = {}
        
    for row in topics_matrix.collect():
        topic_id = row['topic']
        word_indices = row['termIndices'] 
        word_weights = row['termWeights']
        
        top_words = []
        for i, weight in zip(word_indices, word_weights):
            if i < len(vocab):
                word = vocab[i]
                top_words.append((word, float(weight)))
        
        topic_name = interpret_topic_meaning(top_words)
        
        topic_interpretations[str(topic_id)] = {
            'name': topic_name,
            'top_words': top_words,
            'keywords': [word for word, _ in top_words[:5]]
        }
    
    interpretations_json = json.dumps(topic_interpretations, indent=2)
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.defaultFS", "hdfs://namenode:9000")
    fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path("/models/topic_interpretations")
    if fs.exists(path):
        fs.delete(path, True)
    
    sc = spark.sparkContext
    rdd = sc.parallelize([interpretations_json], 1)
    rdd.saveAsTextFile("hdfs://namenode:9000/models/topic_interpretations")
    print("Topic interpretations saved to HDFS")
       
    return topic_interpretations

def train_topics(processed_df, spark):
    print("Starting topic modeling...")
    
    pipeline = create_topic_pipeline()
    print("Training LDA topic model...")
    model = pipeline.fit(processed_df)
    
    lda_model = model.stages[-1]
    cv_model = model.stages[-2]
    transformed_df = model.transform(processed_df)
    log_perplexity = -abs(lda_model.logPerplexity(transformed_df))
    
    extract_topic_interpretations(lda_model, cv_model, spark)
    
    model_path = save_model_to_hdfs("topic_modeling", "LDA", model, log_perplexity, spark)
    print(f"Topic model saved to HDFS: {model_path}")
    
    return model, log_perplexity

def create_clustering_pipeline():
    assembler = VectorAssembler(inputCols=[], outputCol="raw_features", handleInvalid="skip")
    scaler = StandardScaler(inputCol="raw_features", outputCol="clustering_features", withMean=True, withStd=True)
    return [assembler, scaler]

def generate_cluster_interpretations(predictions_df, num_clusters):
    cluster_interpretations = {}
    
    for cluster_id in range(num_clusters):
        cluster_data = predictions_df.filter(F.col("prediction") == cluster_id)
        
        if cluster_data.count() == 0:
            continue
            
        stats = cluster_data.agg(
            F.avg("sentiment_score").alias("avg_sentiment"),
            F.avg("rate_score").alias("avg_rating"),
            F.avg("text_len").alias("avg_length"),
            F.count("*").alias("count")
        ).collect()[0]
        
        avg_sentiment = stats['avg_sentiment'] or 0
        avg_rating = stats['avg_rating'] or 0
        avg_length = stats['avg_length'] or 0
        count = stats['count']
        
        if avg_sentiment < 0.3 and avg_rating < 2.5:
            base_name = "Critical Issues"
            description = "High negative sentiment, urgent complaints requiring attention"
        elif avg_sentiment < 0.5 and avg_rating < 3.5:
            base_name = "Complaints & Concerns"
            description = "General complaints and customer concerns"
        elif avg_sentiment > 0.7 and avg_rating > 4.0:
            base_name = "Satisfied Customers"
            description = "Positive reviews from happy customers"
        elif avg_sentiment > 0.5 and avg_rating > 3.5:
            base_name = "Generally Positive"
            description = "Mostly positive feedback with minor issues"
        else:
            base_name = "Mixed Feedback"
            description = "Balanced sentiment with varied customer experiences"
        
        if avg_length > 200:
            cluster_name = f"{base_name} (Detailed)"
            description += " - Detailed reviews"
        elif avg_length < 50:
            cluster_name = f"{base_name} (Brief)"
            description += " - Brief feedback"
        else:
            cluster_name = base_name
        
        cluster_interpretations[str(cluster_id)] = {
            'name': cluster_name,
            'description': description,
            'characteristics': {
                'avg_sentiment': round(avg_sentiment, 2),
                'avg_rating': round(avg_rating, 2),
                'avg_length': round(avg_length, 0),
                'review_count': count
            }
        }         
    return cluster_interpretations

def save_cluster_interpretations(cluster_interpretations, spark):
    interpretations_json = json.dumps(cluster_interpretations, indent=2)
    
    hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
    hadoop_conf.set("fs.defaultFS", "hdfs://namenode:9000")
    fs = spark.sparkContext._jvm.org.apache.hadoop.fs.FileSystem.get(hadoop_conf)
    path = spark.sparkContext._jvm.org.apache.hadoop.fs.Path("/models/cluster_interpretations")
    if fs.exists(path):
        fs.delete(path, True)
    
    sc = spark.sparkContext
    rdd = sc.parallelize([interpretations_json], 1)
    rdd.saveAsTextFile("hdfs://namenode:9000/models/cluster_interpretations")
    print("Cluster interpretations saved to HDFS")

def train_clustering(df_topics, spark):
    print("Starting clustering analysis...")
    
    features_df, feature_cols = extract_clustering_features(df_topics, "Review", "Summary", "sentiment_label", "Rate_num")
    
    pipeline_base_stages = create_clustering_pipeline()
    pipeline_base_stages[0].setInputCols(feature_cols)

    best_k = 3
    best_score = -1.0
    best_model = None
    
    for k in range(2, 6):
        kmeans = KMeans(featuresCol="clustering_features", k=k, maxIter=20, seed=42)
        temp_pipeline = Pipeline(stages=pipeline_base_stages + [kmeans])
        temp_model = temp_pipeline.fit(features_df)
        
        temp_predictions = temp_model.transform(features_df)
        evaluator = ClusteringEvaluator(featuresCol="clustering_features")
        score = evaluator.evaluate(temp_predictions)
        
        print(f"k={k}: Silhouette Score = {score:.4f}")
        
        if score > best_score:
            best_score = score
            best_k = k
            best_model = temp_model
    
    print(f"Selected k={best_k} with Silhouette Score: {best_score:.4f}")
    
    final_predictions = best_model.transform(features_df)
    interpretations = generate_cluster_interpretations(final_predictions, best_k)
    save_cluster_interpretations(interpretations, spark)
    
    model_path = save_model_to_hdfs("clustering", "KMeans", best_model, best_score, spark)
    print(f"Clustering model saved to HDFS: {model_path}")
    
    return best_model, best_score

def save_model_to_hdfs(model_type, algorithm, model, accuracy, spark):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = f"/models/{model_type}_model"
    full_hdfs_path = f"hdfs://namenode:9000{model_path}"
    
    model.write().overwrite().save(full_hdfs_path)

    accuracy_value = float(accuracy) if accuracy is not None else float('nan')
    
    metadata = spark.createDataFrame([
        (f"{model_type}_model", model_path, accuracy_value, timestamp, algorithm)
    ], ["model_id", "model_path", "accuracy", "timestamp", "algorithm"])
    
    metadata_path = f"hdfs://namenode:9000/models/metadata/{model_type}_metadata"
    metadata.write.mode("overwrite").option("header", "true").csv(metadata_path)
    
    return model_path

def main():
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")
    
    create_hdfs_directories(spark)
    
    local_csv = "/home/jovyan/data/ReviewData/flipkart_reviews_training.csv"
    hdfs_csv = "/data/training/flipkart_reviews_training.csv"
    
    print("Uploading CSV to HDFS...")
    upload_csv_to_hdfs(local_csv, hdfs_csv, spark)
    
    print("Loading data from HDFS...")
    df = load_csv_from_hdfs(hdfs_csv, spark)
    
    spark.catalog.clearCache()

    print("Preprocessing data once...")
    processed_df = preprocess_data(df, is_training=True)
    processed_df = processed_df.cache()
    processed_df.count()

    print("TRAINING SENTIMENT MODEL")
    sentiment_model, acc, f1, precision, recall = train_sentiment_model(spark, processed_df)
            
    print("TRAINING TOPIC MODEL")
    topic_model, topic_score = train_topics(processed_df, spark)
    
    print("PREPARING DATA FOR CLUSTERING")
    df_topics = topic_model.transform(processed_df)
    
    print("TRAINING CLUSTERING MODEL")
    cluster_model, cluster_score = train_clustering(df_topics, spark)
    
    print(f"Training completed - Accuracy: {acc * 100:.2f}%, F1: {f1:.4f}, Topic: {topic_score:.4f}, Clustering: {cluster_score:.4f}")
    
    save_evaluation_results(acc, f1, precision, recall, cluster_score, topic_score)
    
    spark.stop()
    return True

if __name__ == '__main__':
    main()
    print("Training pipeline completed successfully!")