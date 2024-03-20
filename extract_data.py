import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
from pathlib import Path
from azure.storage.blob import BlobServiceClient, BlobClient
from tqdm import tqdm
from io import BytesIO

## Extract data for levels fyi

import pandas as pd
from azure.storage.blob import BlobServiceClient, BlobClient
from tqdm import tqdm
from io import BytesIO
import json

# Initialize the connection to the Azure Blob Storage
connection_string = 'DefaultEndpointsProtocol=https;AccountName=levelfyi;AccountKey=iZYEskIkgkTF43I/hrswZXkuFOhp7FDLrlP2cvZvQhKUBKGB6CCB360M0Hc7S/7P959/yydHuIJd+AStjvkMOw==;EndpointSuffix=core.windows.net'
container_name = 'levelsfyidata'
folder_path = 'responses/20240221'

# Create a BlobServiceClient object
blob_service_client = BlobServiceClient.from_connection_string(connection_string)

# Create a container client
container_client = blob_service_client.get_container_client(container_name)

# List all blobs in the specified folder
blobs_list = container_client.list_blobs(name_starts_with=folder_path)
first_blob = next(iter(blobs_list), None)

levels_fyi_df = pd.DataFrame()
for blob in tqdm(blobs_list):
    blob_client = container_client.get_blob_client(blob)
    blob_data = blob_client.download_blob().readall()
    json_data = json.loads(blob_data)
    blob_df = pd.json_normalize(json_data, record_path=['rows'])
    levels_fyi_df = pd.concat([levels_fyi_df, blob_df], ignore_index=True)

print()
levels_fyi_df.to_csv("levels_fyi.csv", index=False)
print("Finished extracting and saving levels fyi data")
print()


## Extract data for MIT living wages

# Path to the folder containing your JSON files
directory_path = Path("C:/Users/Mudas/Documents/school/Baruch/Data Warehouse/mitlivingwage")

dfs = []

for file_path in tqdm(directory_path.glob("*.json"), desc="Reading JSON files"):
    with open(file_path, 'r') as file:
        data = json.load(file)
    if 'annualSalaryList' in data:
        df = pd.DataFrame(data['annualSalaryList'])
        df['countyName'] = data['countyName']
        df['stateName'] = data['stateName']
        dfs.append(df)
    else:
        print(f"'annualSalaryList' key not found in {file_path.name}")

# Concatenate all DataFrames in the list into a single DataFrame
living_wage_df = pd.concat(dfs, ignore_index=True)
print()

living_wage_df.to_csv("mit_living_wages.csv", index=False)

print("Finished extracting and saving MIT living wages data")
print()

## Extract data for minimum wages


url = "https://minimumwage.com/in-your-state/"
response = requests.get(url)
response.raise_for_status()  # This will raise an error if the request failed

soup = BeautifulSoup(response.text, 'html.parser')
table = soup.find('table')

headers = [header.text for header in table.find_all('th')]

rows = []
for row in table.find_all('tr'):
    columns = row.find_all('td')
    if columns:
        rows.append([column.text for column in columns])


clean_headers = [header.strip() for header in headers[:-1]]
clean_rows = [[col.strip() for col in row[:-1]] for row in rows]

minimum_wage_df = pd.DataFrame(clean_rows, columns=clean_headers)

minimum_wage_df.to_csv("minimum_wage_per_state.csv", index=False)

print("Finished extracting and saving minimum wage data")
print()


## extract start ups data

url = 'https://topstartups.io/startup-salary-equity-database/'
response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

data_rows = []  # Initialize an empty list to hold the data

# Assuming the data is in a table, find the container (e.g., tbody, table)
container = soup.find('table')  # Modify this according to the actual structure

for row in tqdm(container.find_all('tr')):
    cells = row.find_all('td')
    if len(cells) >= 7:  # Assuming you need at least 7 cells for Job Title, Location, Date, and Salary
        job_title = cells[0].text.strip()
        location = cells[6].text.strip()
        date = cells[-1].text.strip()
        salary = cells[1].text.strip()
        num_employees = cells[10].text.strip()
        data_rows.append({'job_title': job_title, 'location': location, 'salary': salary ,"num_employees":num_employees,'date': date})

# Convert the list of dictionaries to a DataFrame
startups_df = pd.DataFrame(data_rows)



startups_df.to_csv("startups.csv", index=False)

print("Finished extracting and saving start ups data")
print()