# Inference Pipeline Script

Write-Host "Starting streaming processor..."
docker exec -d spark-master spark-submit `
   --master spark://spark-master:7077 `
   --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.3.0 `
   /home/jovyan/src/inference_pipeline/spark_streaming.py

Start-Sleep -Seconds 5

Write-Host "Starting dashboard..."
docker exec -d spark-master streamlit run `
    /home/jovyan/src/inference_pipeline/streamlit_dashboard.py `
    --server.port 8501 --server.address 0.0.0.0

Start-Sleep -Seconds 5

Write-Host "Sending reviews to Kafka..."
docker exec spark-master python `
    /home/jovyan/src/inference_pipeline/kafka_review_producer.py `
    /home/jovyan/data/ReviewData/flipkart_reviews_inference.csv 30

Write-Host ""
Write-Host "Check dashboards at:"
Write-Host "   - Streamlit: http://localhost:8501"
Write-Host "   - Kafka UI: http://localhost:8090"
