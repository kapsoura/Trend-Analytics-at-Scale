# Trend Analytics Pipeline Setup Script

Write-Host "==========================================" 
Write-Host "Starting Trend Analytics Pipeline Setup" 
Write-Host "==========================================" 

# Start Docker Compose
Write-Host ""
Write-Host "Step 1: Starting Docker containers..." 
docker compose up -d

# Wait for spark-master to be ready
Write-Host ""
Write-Host "Step 2: Waiting for Spark Master to be ready..." 
Start-Sleep -Seconds 10

# Install dependencies on master node
Write-Host ""
Write-Host "Step 3: Installing dependencies on Spark Master..." 
docker exec spark-master pip install --no-cache-dir -r /home/jovyan/src/requirements.txt

# Install dependencies on all worker nodes
Write-Host ""
Write-Host "Step 4: Installing dependencies on Spark Workers..." 
docker exec spark-worker-1 pip install --no-cache-dir -r /home/jovyan/src/requirements.txt
docker exec spark-worker-2 pip install --no-cache-dir -r /home/jovyan/src/requirements.txt
docker exec spark-worker-3 pip install --no-cache-dir -r /home/jovyan/src/requirements.txt

Write-Host ""
Write-Host "==========================================" 
Write-Host " Setup Complete!" 
Write-Host "==========================================" 
Write-Host ""
Write-Host "Next Steps:" 
Write-Host "1. Run model training:" 
Write-Host "   docker exec spark-master spark-submit --master spark://spark-master:7077 --driver-memory 2g --executor-memory 2g /home/jovyan/src/training_pipeline/sparkml_model_train_hdfs.py" 
Write-Host ""
Write-Host "2. Start inference pipeline:" 
Write-Host "   ./run_pipeline.ps1" 
Write-Host ""
Write-Host "3. Access Dashboards:" 
Write-Host "   - Streamlit: http://localhost:8501" 
Write-Host "   - Spark UI: http://localhost:8080" 
Write-Host "   - Kafka UI: http://localhost:8090" 
Write-Host ""
