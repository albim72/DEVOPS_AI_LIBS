#DAG
from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

def fetch_data():
    print("Pobieranie danych....")

def train_model():
    print("Trenuję model...")

with DAG(
    dag_id="ml_pipeline",
    start_date=datetime(2025,7,23),
    schedule_interval="@daily",
    catchup=False
) as dag:
    task_fetch = PythonOperator(
        task_id="fetch_data",
        python_callable=fetch_data
    )
    task_train = PythonOperator(
        task_id="train_model",
        python_callable=train_model
    )

    task_fetch >> task_train

!airflow webserver --port 8080
