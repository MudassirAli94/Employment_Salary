#!/usr/bin/env python
# coding: utf-8

# ## topstartups.io

# In[5]:


import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm

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

print(startups_df.head(10))

#startups_df.to_csv("startups.csv", index=False)


# ## efinancialcareers

# In[4]:


import requests
import json
import pandas as pd
import re

url = 'https://job-search-api.efinancialcareers.com/v1/efc/jobs/search'
headers = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:123.0) Gecko/20100101 Firefox/123.0',
    'Accept': 'application/json, text/plain, */*',
    'Accept-Language': 'en-US,en;q=0.5',
    'Accept-Encoding': 'gzip, deflate, br',
    'Content-Type': 'application/json',
    'x-api-key': 'zvDFWwKGZ07cpXWV37lpO5MTEzXbHgyL4rKXb39C',
    'Origin': 'https://www.efinancialcareers.com',
    'Connection': 'keep-alive',
    'Referer': 'https://www.efinancialcareers.com/',
    'TE': 'trailers'
}

params = {
    'countryCode2': 'US',
    'radius': 50,
    'radiusUnit': 'mi',
    'page': 1,
    'pageSize': 900,
    'searchId': 'e6b131fe-967c-4014-9e23-c1afb0fc088d',
    'facets': 'locationPath|salaryRange|sectors|employmentType|experienceLevel|workArrangementType|salaryCurrency|minSalary|maxSalary|positionType|postedDate|clientBrandNameFilter',
    'currencyCode': 'USD',
    'culture': 'en',
    'recommendations': 'true',
    'fj': 'false',
    'includeRemote': 'true',
    'includeUnspecifiedSalary': 'true'
}

response = requests.get(url, headers=headers, params=params)
data = response.json()
#print(type(data))
#print(data.keys())


job_data = []

for item in data['data']:
    job_info = {
        'job_title': item['title'],
        'location': item['jobLocation']['city'],
        'salary': item['salary'],
        'employmentType': item['employmentType'],
        #'date': item['expirationDate'],
        #'id': item['id'],
        #'detailsPageUrl': item['detailsPageUrl'],
        #'state': item['jobLocation']['state'],
        #'country': item['jobLocation']['country'],
        'date': item['postedDate'],
        #'workArrangementType': item.get('workArrangementType', None),  
        #'isExternalApplication': item['isExternalApplication'],
        #'summary': item['summary'],
        #'description': item['description']
    }
    job_data.append(job_info)


df = pd.DataFrame(job_data)

# Define a function to remove string values from salary
def extract_salary(salary_str):
    # Use regular expression to find numeric values
    match = re.search(r'\d[\d,]*\d', salary_str)
    if match:
        value = float(match.group().replace(',', ''))
        # Check if the value is less than 1000, implying it's in thousands
        if value < 1000:
            return value * 1000  # Convert to full amount in dollars
        else:
            return value
    else:
        return None
        

df['salary'] = df['salary'].apply(extract_salary)
df = df.dropna(subset=['salary'])

# Define a function to extract year and month from date
def extract_year_month(date_str):
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return year * 100 + month
    else:
        return None

df['date'] = df['date'].apply(extract_year_month)

print(df)

