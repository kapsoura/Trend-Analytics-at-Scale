import csv
import json
import time
import random
from datetime import datetime
from kafka import KafkaProducer

def send_reviews_to_kafka(csv_file, rate_per_minute=30):
    producer = KafkaProducer(
        bootstrap_servers=['broker:9092'],
        value_serializer=lambda v: json.dumps(v).encode('utf-8')
    )
    
    reviews = []
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        reviews = list(reader)
    
    print(f"Loaded {len(reviews)} reviews")
    
    delay = 60.0 / rate_per_minute
    sent_count = 0
    
    for review in reviews:
        message = {
            'review_id': f"review_{sent_count + 1}",
            'timestamp': datetime.now().isoformat(),
            'Product': review.get('Product', ''),
            'Category': review.get('Category', ''),
            'Review': review.get('Review', ''),
            'Summary': review.get('Summary', ''),
            'Rate': review.get('Rate', ''),
            'product_price': review.get('product_price', '')
        }
        
        producer.send('flipkart_product_reviews', value=message)
        sent_count += 1
        
        if sent_count % 50 == 0:
            print(f"Sent {sent_count} reviews")
        
        time.sleep(delay + random.uniform(-0.5, 0.5))
    
    producer.flush()
    producer.close()
    print(f"Finished sending {sent_count} reviews")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python kafka_review_producer.py <csv_file> [rate_per_minute]")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    rate = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    
    send_reviews_to_kafka(csv_file, rate)