# Installation Guide

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- PowerShell (Windows)

## Quick Start

1. **Clone the repository**
   ```bash
   gh repo clone kapsoura/Trend-Analytics-at-Scale
   cd Trend-Analytics-at-Scale
   ```

2. **Run the startup script**
   ```bash
   cd Setup
   ./startup.ps1
   ```
   This will start all containers and install dependencies inside docker automatically.


## Training Models

Run the training pipeline:
```bash
docker exec spark-master spark-submit \
  --master spark://spark-master:7077 \
  --driver-memory 2g \
  --executor-memory 2g \
  /home/jovyan/src/training_pipeline/sparkml_model_train_hdfs.py
```

Models are saved to HDFS

## Running Inference Pipeline

Start the streaming inference system:
```bash
cd Setup
./run_pipeline.ps1
```

Access dashboards:
- Streamlit Dashboard: http://localhost:8501
- Kafka UI: http://localhost:8090
- Spark Master: http://localhost:8080

## Stopping Services

```bash
cd Setup
docker-compose -f Dockercompose.yaml down
```