from google.cloud import storage
from datetime import datetime
from google.cloud import bigquery
import pandas as pd
from io import StringIO
from google.oauth2 import service_account
import pandas_gbq

def upload_dataframe_to_gcs(bucket_name, df, destination_blob_name, project_id):
    now = datetime.now()

    formatted_time = now.strftime("%Y%m%d%H%M%S")
    destination_blob_name_with_timestamp = f"{destination_blob_name}_{formatted_time}.csv"

    csv_string = df.to_csv(index=False)
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name_with_timestamp)
    blob.upload_from_string(csv_string, content_type='text/csv')

    print(f"DataFrame uploaded to {destination_blob_name_with_timestamp} in bucket {bucket_name}.")


def read_csv_from_gcs(bucket_name, blob_name):
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    data = blob.download_as_text()
    df = pd.read_csv(StringIO(data))

    return df



def upload_table_to_bq(job_config, df, project_id, dataset_name,table_name):

    client = bigquery.Client(project=project_id)
    dataset_id = f'{project_id}.{dataset_name}'
    table_id = f'{dataset_id}.{table_name}'
    dataset = bigquery.Dataset(dataset_id)
    client.create_dataset(dataset, exists_ok=True)

    job = client.load_table_from_dataframe(df, table_id, job_config=job_config)
    print(job.result())
    print()
    print("Table uploaded to BigQuery.")

# job config example:

# job_config = bigquery.LoadJobConfig(
#     schema=[
#         # Define your schema here. Example:
#         bigquery.SchemaField("name", "STRING"),
#         bigquery.SchemaField("age", "INTEGER"),
#     ],
#     write_disposition="WRITE_TRUNCATE",  # Adjust based on your needs
# )

def read_table_from_bq(query,project_id):
    credentials = service_account.Credentials.from_service_account_file(
        r"C:\Users\Mudas\Documents\school\Baruch\Data Warehouse\data\Term Project.json")
    # Read the data from BigQuery into a pandas DataFrame
    return pandas_gbq.read_gbq(query, project_id=project_id, credentials=credentials)


## to make credentials open goiogle cloud SDK shell and type the following:
## gcloud iam service-accounts keys create "<location of key>" --iam-account group9@dw-group-project.iam.gserviceaccount.com

