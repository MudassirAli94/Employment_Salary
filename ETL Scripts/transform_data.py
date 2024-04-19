import numpy as np
import pandas as pd
import re
from gcp_functions import read_csv_from_gcs, insert_dataframe_to_bigquery, create_bigquery_schema
import json
import uuid



## GCP configuration

with open('config.json') as config_file:
    config = json.load(config_file)

YOUR_BUCKET_NAME = config["bucket_name"]
PROJECT_ID = config["project_id"]

## clean and transform levels_fyi_data

state_abbreviations = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR", "California": "CA",
    "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE", "Florida": "FL", "Georgia": "GA",
    "Hawaii": "HI", "Idaho": "ID", "Illinois": "IL", "Indiana": "IN", "Iowa": "IA",
    "Kansas": "KS", "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV", "New Hampshire": "NH",
    "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY", "North Carolina": "NC",
    "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK", "Oregon": "OR", "Pennsylvania": "PA",
    "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", "Tennessee": "TN",
    "Texas": "TX", "Utah": "UT", "Vermont": "VT", "Virginia": "VA", "Washington": "WA",
    "West Virginia": "WV", "Wisconsin": "WI", "Wyoming": "WY",
    "District of Columbia": "DC", "Puerto Rico": "PR", "Remote":"Remote"
}

levels_df = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name="2024-04-15/levels_fyi_20240415170012.csv")

levels_df['offerDate'] = levels_df['offerDate'].str.slice(start=0, stop=24)
# Now, convert the 'offerDate' column to datetime without specifying a format
levels_df['offerDate'] = pd.to_datetime(levels_df['offerDate'])
# Format the datetime as 'YYYYMMDD' string and then convert to integer
levels_df['offerDate'] = levels_df['offerDate'].dt.strftime('%Y%m%d').astype(int)

## getting short hand for state

levels_df["state_short"] = levels_df.location.apply(lambda i: str(i).split(', ')[-1])
levels_df["state"] = levels_df["state_short"].map({abbrev: state for state, abbrev in state_abbreviations.items()})
levels_df["city"] = levels_df.location.apply(lambda i: str(i).split(', ')[0])

## only interested in levels in USA

levels_df = levels_df[levels_df.state_short.isnull() != True]
levels_df = levels_df[levels_df.state_short.isin(state_abbreviations.values())]

## replace the brackets and quotes in the tags column

levels_df["tags"] = levels_df["tags"].apply(lambda i: str(i).replace("[",""))\
.apply(lambda i: str(i).replace("]","")).apply(lambda i: str(i).replace("'",""))\
    .apply(lambda i: str(i).replace(",","")).apply(lambda i: str(i).replace("-",""))\
    .apply(lambda i: np.nan if i == 'nan' else i)

## dropping columns that are more than 50% missing

missing_percentages = levels_df.isnull().mean() * 100
columns_to_drop = missing_percentages[missing_percentages > 50].index

levels_df = levels_df.drop(columns=columns_to_drop)
levels_df = levels_df.drop(columns=["location","companyInfo.slug","exchangeRate","companyInfo.registered",
                                    "baseSalaryCurrency","countryId","cityId",'compPerspective',"totalCompensation"])


## rename column to snake case
def to_snake_case(column_name):
    # Replace dots with underscores
    column_name = column_name.replace('.', '_')
    # Insert underscores before capital letters and convert to lowercase
    return ''.join(['_' + i.lower() if i.isupper() else i for i in column_name]).lstrip('_')

# Apply the conversion function to each column name in the list
snake_case_columns = [to_snake_case(column) for column in levels_df.columns]

levels_df.columns = snake_case_columns



## Rearrange columns

levels_df = levels_df[['uuid','state','state_short', 'city', 'title', 'job_family', 'level', 'focus_tag',
       'years_of_experience', 'years_at_company', 'years_at_level',
       'offer_date', 'work_arrangement', 'dma_id', 'base_salary', 'company_info_icon', 'company_info_name']]

levels_df = levels_df.rename(columns = {"base_salary":"salary","uuid":"job_id"})
levels_df = levels_df.drop_duplicates(subset = ["job_id"])

## convert floats to int

for n in ["salary","dma_id"]:
    levels_df[n] = levels_df[n].astype(int)


unique_names = levels_df['company_info_name'].unique()

levels_df["city"] = levels_df["city"].apply(lambda i: i.replace("West McLean","McLean"))

for n in ["years_at_company","years_at_level","years_of_experience"]:
    levels_df[n] = levels_df[n].apply(lambda i: str(i).replace("0-1","0"))
    levels_df[n] = levels_df[n].apply(lambda i: str(i).replace("2-4", "3"))
    levels_df[n] = levels_df[n].apply(lambda i: str(i).replace("5-10", "7"))
    levels_df[n] = levels_df[n].apply(lambda i: str(i).replace("+",""))
    levels_df[n] = levels_df[n].apply(lambda i: str(i).replace("10-May", ""))
    levels_df[n] = levels_df[n].apply(lambda i: str(i).replace("4-Feb", ""))
    levels_df = levels_df[~levels_df[n].str.contains('-', na=False)]
    levels_df[n] = levels_df[n].astype(float)


print(levels_df.head())

## clean and transform mit living wages data

living_wage_df = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name="2024-04-15/mit_living_wages_20240415170017.csv")

living_wage_df = pd.read_csv("../mit_living_wages.csv")

living_wage_df.typicalAnnualSalary = living_wage_df.typicalAnnualSalary.apply(lambda i: str(i).replace("$","")).apply(lambda i: str(i).replace(",",""))
living_wage_df.typicalAnnualSalary = living_wage_df.typicalAnnualSalary.astype(int)



cols_list = ["occupational_area","annual_salary","location_name","state"]
living_wage_df.columns = cols_list

living_wage_df["occupational_area"] = living_wage_df["occupational_area"].apply(lambda i: i.replace('"',''))

living_wage_df["location_name"] = living_wage_df["location_name"].apply(lambda i: str(i).split(",")[0])

living_wage_df['state_short'] = living_wage_df['state'].map(state_abbreviations)
living_wage_df["location_name"] = living_wage_df["location_name"].apply(lambda i: i.replace(" County",""))
living_wage_df["location_name"] = living_wage_df["location_name"].apply(lambda i: i.replace(" city",""))
living_wage_df["location_name"] = living_wage_df["location_name"].apply(lambda i: i.replace(" Borough",""))
living_wage_df["location_name"] = living_wage_df["location_name"].apply(lambda i: i.replace("New York-Newark-Jersey City, NY","Newark"))
living_wage_df["location_name"] = living_wage_df["location_name"].apply(lambda i: i.split("-")[0])

living_wage_df = living_wage_df.rename(columns={"annual_salary":"salary" ,"location_name":"county"})
living_wage_df_newark = living_wage_df[living_wage_df["county"] == "Newark"]
living_wage_df_newark["county"] = "Jersey"
living_wage_df = pd.concat([living_wage_df, living_wage_df_newark])
living_wage_df = living_wage_df.sort_values(by = ["state","county"])

#print(living_wage_df.head())

category_map = {
    'Marketing Operations': 'Business & Financial Operations',
    'Software Engineer': 'Computer & Mathematical',
    'Mechanical Engineer': 'Architecture & Engineering',
    'Program Manager': 'Management',
    'Business Analyst': 'Business & Financial Operations',
    'Software Engineering Manager': 'Computer & Mathematical',
    'Recruiter': 'Business & Financial Operations',
    'Geological Engineer': 'Architecture & Engineering',
    'Accountant': 'Business & Financial Operations',
    'Project Manager': 'Management',
    'Business Development': 'Business & Financial Operations',
    'Technical Program Manager': 'Computer & Mathematical',
    'Product Designer': 'Arts, Design, Entertainment, Sports, & Media',
    'Financial Analyst': 'Business & Financial Operations',
    'Sales': 'Sales & Related',
    'Data Science Manager': 'Computer & Mathematical',
    'Human Resources': 'Business & Financial Operations',
    'Product Design Manager': 'Arts, Design, Entertainment, Sports, & Media',
    'Solution Architect': 'Computer & Mathematical',
    'Venture Capitalist': 'Business & Financial Operations',
    'Product Manager': 'Management',
    'Biomedical Engineer': 'Architecture & Engineering',
    'Administrative Assistant': 'Office & Administrative Support',
    'Technical Writer': 'Arts, Design, Entertainment, Sports, & Media',
    'Civil Engineer': 'Architecture & Engineering',
    'Chief of Staff': 'Management',
    'Management Consultant': 'Management',
    'Legal': 'Legal',
    'Hardware Engineer': 'Architecture & Engineering',
    'Copywriter': 'Arts, Design, Entertainment, Sports, & Media',
    'Marketing': 'Business & Financial Operations',
    'Customer Service': 'Office & Administrative Support',
    'Data Scientist': 'Computer & Mathematical',
    'Security Analyst': 'Computer & Mathematical',
    'Information Technologist': 'Computer & Mathematical',
    'Industrial Designer': 'Arts, Design, Entertainment, Sports, & Media',
    'Founder': 'Management',
    'Fashion Designer': 'Arts, Design, Entertainment, Sports, & Media',
    'Investment Banker': 'Business & Financial Operations'
}

levels_df['occupational_area'] = levels_df["job_family"].map(category_map)

living_wage_df = living_wage_df.rename(columns = {"salary":"mit_estimated_salary"})


## clean and transform minimum wage data

minimum_wage_df = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name="2024-04-15/minimum_wage_per_state_20240415170019.csv")

minimum_wage_df = pd.read_csv("../minimum_wage_per_state.csv")

cols_list = minimum_wage_df.columns.tolist()
cols_list = [col.replace(" ", "_").lower() for col in cols_list]
minimum_wage_df.columns = cols_list

for n in ["minimum_wage", "tipped_wage"]:

    minimum_wage_df[n] = minimum_wage_df[n].apply(lambda i: i.replace("$",""))
    minimum_wage_df[n] = minimum_wage_df[n].astype(float)


minimum_wage_df['state_short'] = minimum_wage_df['state'].map(state_abbreviations)

minimum_wage_df = minimum_wage_df[["state", "state_short", "minimum_wage", "tipped_wage"]]

## INSERT DATA INGESTION TO DATAWARE HOUSE CODE HERE

print(minimum_wage_df.head())


## clean and transform start up jobs data

start_ups_df = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name="2024-04-15/startups_jobs_20240415170023.csv")


start_ups_df["num_employees"] = start_ups_df["num_employees"].apply(lambda i: str(i).split(" ")[0])\
    .apply(lambda i: str(i).split("-")[-1]).apply(lambda i: str(i).replace("+",""))
start_ups_df["num_employees"] = start_ups_df["num_employees"].astype(int)

start_ups_df["date"] = start_ups_df["date"].apply(lambda i: str(i).replace("/",""))
start_ups_df["date"] = start_ups_df["date"].astype(int)

start_ups_df['salary'] = start_ups_df['salary'].str.replace(r"\+.*", "", regex=True)
start_ups_df['salary'] = start_ups_df['salary'].str.replace("$","")
start_ups_df['salary'] = start_ups_df['salary'].str.replace(",","")
start_ups_df['salary'] = start_ups_df['salary'].astype(int)

location_mapping = {
    'San Francisco': 'CA',
    'San Francisco Bay Area': 'CA',
    'New York City': 'NY',
    'New York': 'NY',
    'NYC': 'NY',
    # Add more mappings as needed
}


def map_location_to_state(location, state_abbreviations):
    # Simplified mapping for cities to states, expand as needed
    city_to_state = {
        "San Francisco": "California", "San Francisco Bay Area": "California",
        "New York City": "New York", "Los Angeles": "California", "Seattle": "Washington",
        "Austin": "Texas", "Boston": "Massachusetts", "Chicago": "Illinois", "Dallas": "Texas",
        "Houston": "Texas", "Miami": "Florida", "San Diego": "California", "Washington DC": "District of Columbia",
        "Remote": "Remote", "Remote US": "Remote"
        # Add more mappings as necessary
    }
    # Handle generic 'Remote' cases or specific 'Remote US' cases by returning None or a specific value
    if 'UAE' in location or 'Estonia' in location or 'Remote Switzerland' in location:
        return None

    # Map city names to states, and then states to abbreviations
    if location in city_to_state:
        state_name = city_to_state[location]
        return state_abbreviations.get(state_name, None)

    # Directly map known state names to abbreviations
    return state_abbreviations.get(location, None)


start_ups_df['state_short'] = start_ups_df['location'].apply(lambda x: map_location_to_state(x, state_abbreviations))

# Drop rows with None in 'location' if needed

start_ups_df = start_ups_df[start_ups_df["state_short"].isnull() != True]

start_ups_df["location"] = start_ups_df["location"].replace(["San Francisco Bay Area","New York City","Remote US"],["San Francisco","New York","Remote"])

start_ups_df = start_ups_df.rename(columns = {"location":"city"})

start_ups_df = start_ups_df[["job_title","city","state_short","salary","num_employees","date"]]

## INSERT DATA INGESTION TO DATAWARE HOUSE CODE HERE

print(start_ups_df.head())


## transform census data


census_df = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name="2024-04-15/2020_census_data_20240415170027.csv")


census_df["county"] = census_df["NAME"].apply(lambda i: i.split(",")[0])
census_df["county"] = census_df["county"].apply(lambda i:str(i).replace('"',''))
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" County",""))
census_df["state"] = census_df["NAME"].apply(lambda i: i.split(", ")[1])
census_df["total_population"] = census_df["P1_001N"].astype(int)
census_df = census_df[["county_geo_id","county", "state", "total_population"]]
census_df["state_short"] = census_df["state"].map(state_abbreviations)
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" Municipio",""))
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" Municipality",""))
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" City and Borough",""))
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" city",""))
census_df["county_geo_id"] = census_df["county_geo_id"].astype(str).str.zfill(5)

census_df = census_df[["county_geo_id",'state','state_short','county','total_population']]

print(census_df.head(10))


## INSERT DATA INGESTION TO DATAWARE HOUSE CODE HERE



print(census_df.head())

## transform gazetter data

gaz_df = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name="2024-04-15/gazetteer_data_20240415170051.csv")

gaz_df = pd.read_csv("gazetteer_data.csv", dtype=str)
gaz_df = gaz_df.rename(columns = {"state":"state_short"})
gaz_df["state"] = gaz_df["state_short"].map({abbrev: state for state, abbrev in state_abbreviations.items()})
ny_gaz_df = gaz_df[(gaz_df["county"] == "New York") & (gaz_df["name"]== "New York")]
gaz_df = gaz_df[~gaz_df.county.isin(state_abbreviations.keys())]
gaz_df = pd.concat([gaz_df, ny_gaz_df])
gaz_df = gaz_df.sort_values(by = ["state","county","name"])
gaz_df['county'] = gaz_df['county'].astype(str)
gaz_df = gaz_df[~gaz_df['county'].str.match(r'^\s*\d{5}\s*$')]
gaz_df = gaz_df[~gaz_df['county'].str.strip().str.isdigit()]
gaz_df = gaz_df[gaz_df['county'].str.lower() != 'nan']
gaz_df['county'] = gaz_df['county'].str.strip()
gaz_df["county"] = gaz_df["county"].apply(lambda i: i.replace(" County",""))
gaz_df.loc[(gaz_df['state'] == 'District of Columbia') & (gaz_df['name'] == 'Washington'), 'county'] = "District of Columbia"

gaz_df = gaz_df[['state','state_short','county','name','type_of_place','geo_id','ansi_code']]
gaz_df = gaz_df.rename(columns = {"name":"city"})


## INSERT DATA INGESTION TO DATAWARE HOUSE CODE HERE
print(gaz_df.head())
## transform DMA data

dma_df = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name="2024-04-15/dma_data_20240415170025.csv")

dma_df = pd.read_csv("dma.csv")



dma_df = dma_df.rename(columns = {"Designated Market Area (DMA)":"location_name","Rank":"rank","TV Homes":"tv_homes","% of US":"percent_of_united_states","DMA Code":"dma_id"})

dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.split(",")[0])
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.split(" (")[0])
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.split("(")[0])
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.split("-")[0])
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.split(" &")[0])
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.split("&")[0])
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace(" City",""))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("Ft.","Fort"))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("Sacramnto","Sacramento"))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("SantaBarbra","Santa Barbara"))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("Idaho Fals","Idaho Falls"))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("Honolulu","Honolulu"))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("Rochestr","Rochester"))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("Greenvll","Greenville"))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("Wilkes Barre","Wilkes-Barre"))
dma_df["location_name"] = dma_df["location_name"].apply(lambda i: i.replace("St Joseph","St. Joseph"))

dma_df = dma_df.sort_values(by="rank")

## INSERT DATA INGESTION TO DATAWARE HOUSE CODE HERE

print(dma_df.head())


## transform efinancial data

efinancial_df = read_csv_from_gcs(bucket_name=YOUR_BUCKET_NAME, file_name="2024-04-15/efinancial_jobs_20240415170055.csv")

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
        return 0


efinancial_df['salary'] = efinancial_df['salary'].apply(extract_salary)
efinancial_df = efinancial_df.dropna(subset=['salary'])


# Define a function to extract year and month from date
def extract_year_month(date_str):
    match = re.match(r'(\d{4})-(\d{2})-(\d{2})', date_str)
    if match:
        year = int(match.group(1))
        month = int(match.group(2))
        return year * 100 + month
    else:
        return None


efinancial_df['date'] = efinancial_df['date'].apply(extract_year_month)


def extract_state_abbreviation(state_str):
    state_name = str(state_str).split(', ')[-1]
    return state_abbreviations.get(state_name)


# Apply the function to the 'state' column to create a new column 'state_short'
efinancial_df['state_short'] = efinancial_df['state'].apply(extract_state_abbreviation)
df_job_data = efinancial_df[efinancial_df['state_short'].isin(state_abbreviations.values())]
df_job_data = df_job_data.drop(columns=['state'])
df_job_data.insert(loc=2, column='state_short', value=df_job_data.pop('state_short'))

df_job_data.job_title = df_job_data.job_title.apply(lambda i: i.replace('"',''))
df_job_data.job_title = df_job_data.job_title.apply(lambda i: i.replace("2025 ", ""))


## INSERT DATA INGESTION TO DATAWARE HOUSE CODE HERE
print(df_job_data)


## getting fact table

levels_facts_df = levels_df[["job_id", "dma_id", "state", "state_short","city","salary","years_of_experience","years_at_company","years_at_level", "occupational_area"]].copy()
minimum_facts_df = minimum_wage_df[["state","minimum_wage","tipped_wage"]].copy()
county_facts_df = census_df[["county_geo_id","state","state_short","county","total_population"]].copy()
county_facts_df = county_facts_df.merge(gaz_df[["county","state_short","city","type_of_place"]], on = ["county","state_short"], how = "left")
print(county_facts_df.head())

living_wage_fact_df = living_wage_df.merge(gaz_df[["county","state_short","city","type_of_place"]], on = ["county","state_short"], how = "left")
levels_facts_df = levels_facts_df.merge(living_wage_fact_df[["occupational_area","city", "state_short","mit_estimated_salary"]]\
                                        , on = ["occupational_area","city","state_short"], how = "left")
levels_facts_df.drop_duplicates(subset = ["job_id"], inplace=True)
levels_facts_df = levels_facts_df.rename(columns = {"mit_estimated_salary":"mit_estimated_baseline_salary"})

##merge with gaz_df to get city_town_geo_id
levels_facts_df = levels_facts_df.merge(gaz_df[["state","city","geo_id"]], on = ["state","city"], how = "left")
levels_facts_df = levels_facts_df.rename(columns = {"geo_id":"city_town_geo_id"})

## merge with dma_df
levels_facts_df = levels_facts_df.merge(dma_df[["dma_id","rank","tv_homes","percent_of_united_states"]], on = "dma_id", how = "left")
## merge with minimum_wage_df
levels_facts_df = levels_facts_df.merge(minimum_wage_df[["state","minimum_wage","tipped_wage"]], on = "state", how = "left")
## merge with county_facts_df
levels_facts_df = levels_facts_df.merge(county_facts_df[["state","county","city","county_geo_id","total_population"]], on = ["city","state"], how = "left")
## getting average total_population because of many counties rolling up to one city
mean_population_df = levels_facts_df.groupby("job_id")["total_population"].mean().reset_index()
levels_facts_df = levels_facts_df.drop(columns=["total_population"])
levels_facts_df = levels_facts_df.merge(mean_population_df, on = "job_id", how = "left")
levels_facts_df = levels_facts_df.rename(columns = {"total_population":"county_avg_total_population"})
levels_facts_df["county_avg_total_population"] = round(levels_facts_df["county_avg_total_population"],0)

levels_facts_df = levels_facts_df.drop(columns=["state","state_short","city","occupational_area", "county"])
levels_facts_df = levels_facts_df.drop_duplicates(subset = ["job_id"])

levels_facts_df = levels_facts_df.drop_duplicates(subset = ["job_id"])

facts_df = levels_facts_df.copy()



## get dma dimension

dim_dma_df = dma_df[["dma_id","location_name"]].copy()
print(dim_dma_df.head())

## get dim location dimension

dim_location_df = gaz_df[["geo_id","state","state_short","county","city","type_of_place"]].copy()
dim_location_df = dim_location_df.merge(census_df[["county_geo_id","state","county"]], on = ["county","state"], how = "left")
dim_location_df = dim_location_df.rename(columns = {"geo_id":"city_town_geo_id", "city":"location_name"})

## get dim jobs dimension

dim_jobs_df = efinancial_df[["job_title","city","state_short","salary"]].copy()
dim_jobs_df = pd.concat([dim_jobs_df, start_ups_df[["job_title","city","state_short","salary"]]])

## creating a primary key for the dim_jobs_df
def generate_uuid():
    return str(uuid.uuid4())

# Apply the generate_uuid function to a new column 'job_id'
dim_jobs_df['job_id'] = dim_jobs_df.apply(lambda x: generate_uuid(), axis=1)

occupational_area_map = {
    'Marketing Operations': 'Business & Financial Operations',
    'Software Engineer': 'Computer & Mathematical',
    'Mechanical Engineer': 'Architecture & Engineering',
    'Program Manager': 'Management',
    'Business Analyst': 'Business & Financial Operations',
    'Software Engineering Manager': 'Computer & Mathematical',
    'Recruiter': 'Human Resources',
    'Geological Engineer': 'Architecture & Engineering',
    'Accountant': 'Business & Financial Operations',
    'Project Manager': 'Management',
    'Business Development': 'Sales & Related',
    'Technical Program Manager': 'Computer & Mathematical',
    'Product Designer': 'Arts, Design, Entertainment, Sports, & Media',
    'Financial Analyst': 'Business & Financial Operations',
    'Sales': 'Sales & Related',
    'Data Science Manager': 'Computer & Mathematical',
    'Human Resources': 'Human Resources',
    'Product Design Manager': 'Arts, Design, Entertainment, Sports, & Media',
    'Solution Architect': 'Computer & Mathematical',
    'Venture Capitalist': 'Business & Financial Operations',
    'Product Manager': 'Management',
    'Biomedical Engineer': 'Architecture & Engineering',
    'Administrative Assistant': 'Office & Administrative Support',
    'Technical Writer': 'Arts, Design, Entertainment, Sports, & Media',
    'Civil Engineer': 'Architecture & Engineering',
    'Chief of Staff': 'Management',
    'Management Consultant': 'Management',
    'Legal': 'Legal',
    'Hardware Engineer': 'Computer & Mathematical',
    'Copywriter': 'Arts, Design, Entertainment, Sports, & Media',
    'Marketing': 'Business & Financial Operations',
    'Customer Service': 'Office & Administrative Support',
    'Data Scientist': 'Computer & Mathematical',
    'Security Analyst': 'Computer & Mathematical',
    'Information Technologist': 'Computer & Mathematical',
    'Industrial Designer': 'Arts, Design, Entertainment, Sports, & Media',
    'Founder': 'Management',
    'Fashion Designer': 'Arts, Design, Entertainment, Sports, & Media',
    'Investment Banker': 'Business & Financial Operations'
}

# Function to match job titles to job family and occupational area

def match_job(title):
    title_lower = title.lower()

    # Using regex to match full words or specific phrases to avoid partial matches
    if re.search(r'\b(cio|ceo|cto|coo|cfo|chief executive officer|chief technology officer|chief operating officer|chief)\b', title_lower):
        return 'Chief of Staff', 'Management'

    if re.search(r'\b(president|vice president|vp|avp|svp|director)\b', title_lower):
        return 'Program Manager', 'Management'

    if re.search(r'\b("lead software|lead devops|lead mobile|lead ios|lead frontend|lead backend|lead machine learning engineer")\b', title_lower):
        return 'Software Engineering Manager', 'Computer & Mathematical'

    if re.search(r'\b(lead data scientist|head of data science|manager, data sci|senior manager, data sci)\b', title_lower):
        return 'Data Science Manager', 'Computer & Mathematical'

    if re.search(r'\b(data science|data scientist)\b', title_lower):
        return 'Data Scientist', 'Computer & Mathematical'

    if re.search(r'\b(product manager|head of product|manager, product|product management|director of product|vp of product|product lead|lead product)\b', title_lower):
        return 'Product Manager', 'Management'

    if re.search(r'\b(business operation|bizops|business development|business intelligence)\b', title_lower):
        return 'Business & Financial Operations', 'Business & Financial Operations'

    if re.search(r'\b(machine learning engineer|analyst|analytics|quantitative|data engineer|computer vision|ux|principal ai|database)\b', title_lower):
        return 'Computer & Mathematical', 'Computer & Mathematical'

    if re.search(r'\b(wealth|investment|investments|banker|financial|finance)\b', title_lower):
        return 'Financial Analyst', 'Business & Financial Operations'

    if re.search(r'\b(product|content designer|brand designer)\b', title_lower):
        return 'Product Designer', 'Arts, Design, Entertainment, Sports, & Media'

    if re.search(r'\b(devops|developer|software|frontend|backend|front end|back end|fullstack|full stack|full-stack|principal engineer|systems engineer|support engineer|firmware engineer|solutions engineer)\b', title_lower):
        return 'Software Engineer', 'Computer & Mathematical'

    if re.search(r'\b(representative|customer service|customer experience|customer support)\b', title_lower):
        return 'Customer Service', 'Office & Administrative Support'

    if re.search(r'\b(actuary)\b', title_lower):
        return 'Financial Analyst', 'Business & Financial Operations'

    if re.search(r'\b(teller|cashier)\b', title_lower):
        return 'Customer Service', 'Office & Administrative Support'

    if re.search(r'\b(legal|counsel|attorney|lawyer|paralegal)\b', title_lower):
        return 'Legal', 'Legal'

    if re.search(r'\b(mechanical engineer|civil engineer|industrial engineer)\b', title_lower):
        return 'Mechanical Engineer', 'Architecture & Engineering'

    if re.search(r'\b(hardware engineer|hardware systems architect)\b', title_lower):
        return 'Hardware Engineer', 'Architecture & Engineering'

    if re.search(r'\b(underwriter)\b', title_lower):
        return 'Copywriter', 'Arts, Design, Entertainment, Sports, & Media'

    if re.search(r'\b(security engineer)\b', title_lower):
        return 'Security Analyst', 'Computer & Mathematical'

    if re.search(r'\b(systems engineer)\b', title_lower):
        return 'Information Technologist', 'Computer & Mathematical'

    if re.search(r'\b(solutions architect|solution architect)\b', title_lower):
        return 'Solution Architect', 'Computer & Mathematical'

    if re.search(r'\b(data governance|infrastructure engineer)\b', title_lower):
        return 'Information Technologist', 'Computer & Mathematical'

    if re.search(r'\b(sales engineer)\b', title_lower):
        return 'Sales', 'Sales & Related'

    if re.search(r'\b(manager|lead|head|supervisor|coordinator|executive)\b', title_lower):
        return 'Management', 'Management'

    return "Other", "Other"  # Fallback if no match is found


# Apply the final customer service specific matching function to the DataFrame
dim_jobs_df['job_family'], dim_jobs_df['occupational_area'] = zip(*dim_jobs_df['job_title'].apply(match_job))

# Display the updated DataFrame


dim_jobs_df["state"] = dim_jobs_df["state_short"].map({abbrev: state for state, abbrev in state_abbreviations.items()})

dim_jobs_df["city"] = dim_jobs_df["city"].apply(lambda i: i.replace(" City",""))
dim_jobs_df["city"] = dim_jobs_df["city"].apply(lambda i: i.replace("Manhattan","New York"))

## get minimum wage per state for dim_jobs_df
dim_jobs_df = dim_jobs_df.merge(minimum_wage_df[["state","minimum_wage","tipped_wage"]], on = "state", how = "left")
## get DMA data for dim_jobs_df
dim_jobs_df = dim_jobs_df.merge(dma_df.rename(columns = {"location_name":"city"}), on = "city", how = "left")
## get county data for dim_jobs_df
dim_jobs_df = dim_jobs_df.merge(county_facts_df[["state","county","city","county_geo_id","total_population"]], on = ["city","state"], how = "left")
## get city_town_geo_id for dim_jobs_df
dim_jobs_df = dim_jobs_df.merge(gaz_df[["state","city","geo_id"]], on = ["state","city"], how = "left")
dim_jobs_df = dim_jobs_df.rename(columns = {"geo_id":"city_town_geo_id"})
## get average total_population for dim_jobs_df because many counties roll up to one city
mean_population_df = dim_jobs_df.groupby("job_id")["total_population"].mean().reset_index()
dim_jobs_df = dim_jobs_df.drop(columns=["total_population"])
dim_jobs_df = dim_jobs_df.merge(mean_population_df, on = "job_id", how = "left")
## merge living wages to get mit estimated salary
dim_jobs_df = dim_jobs_df.merge(living_wage_df[["occupational_area","county","mit_estimated_salary"]], on = ["occupational_area","county"], how = "left")
## get average mit estimated salary because many counties roll up to one city
mean_mit_salary_df = dim_jobs_df.groupby("job_id")["mit_estimated_salary"].mean().reset_index()
dim_jobs_df = dim_jobs_df.drop(columns=["mit_estimated_salary"])
dim_jobs_df = dim_jobs_df.merge(mean_mit_salary_df, on = "job_id", how = "left")
dim_jobs_df = dim_jobs_df.drop_duplicates(subset = ["job_id"])

## ingesting dim_jobs_facts to our facts_df
dim_jobs_facts_df = dim_jobs_df[['job_id','city','state','state_short','job_title','job_family','occupational_area','salary','minimum_wage','tipped_wage','dma_id','total_population','mit_estimated_salary',"rank","tv_homes","percent_of_united_states"]].copy()
dim_jobs_facts_df = dim_jobs_facts_df.rename(columns = {"mit_estimated_salary":"mit_estimated_baseline_salary", "total_population":"county_avg_total_population"})
dim_jobs_facts_df  = dim_jobs_facts_df.drop(columns = ["city","state","state_short","job_title", "job_family","occupational_area"])

facts_df = pd.concat([facts_df, dim_jobs_facts_df]).copy()

print(facts_df.head())





## now making dim_jobs table

dim_jobs_df1 = dim_jobs_df[["job_id","city","state","state_short","job_title","job_family","occupational_area"]].copy()


levels_dim_jobs_df = levels_df[["job_id","company_info_name","company_info_icon","state","state_short","city","title","job_family","occupational_area"]].copy()
levels_dim_jobs_df = levels_dim_jobs_df.rename(columns={"title":"job_title","company_info_name":"company_name","company_info_icon":"company_icon"})

final_dims_jobs_df = pd.concat([dim_jobs_df1, levels_dim_jobs_df]).copy()

assert final_dims_jobs_df.job_id.isin(facts_df.job_id).sum() == final_dims_jobs_df.shape[0], "All jobs in dim_job table are not in facts table"
assert facts_df.job_id.isin(final_dims_jobs_df.job_id).sum() == facts_df.shape[0], "All jobs in facts table are not in dim_job table"
print()

## Ingest tables into datawarehouse

## create schema

# Column order for the dim_jobs table
dim_jobs_columns = [
    'job_id', 'company_name', 'company_icon', 'state', 'state_short',
    'city', 'job_title', 'job_family', 'occupational_area'
]

# Column order for the facts_jobs table
facts_jobs_columns = [
    'job_id', 'dma_id', 'city_town_geo_id', 'county_geo_id', 'salary',
    'mit_estimated_baseline_salary', 'years_of_experience', 'years_at_level',
    'county_avg_total_population', 'minimum_wage', 'tipped_wage', 'rank',
    'tv_homes', 'percent_of_united_states'
]

# Column order for the dim_location table
dim_location_columns = [
    'city_town_geo_id', 'county_geo_id', 'state', 'state_short', 'county',
    'location_name', 'type_of_place'
]

# Column order for the dim_dma table
dim_dma_columns = [
    'dma_id', 'location_name'
]

# Assuming final_dims_jobs_df, facts_df, dma_df, and dim_location_df are your DataFrames
final_dims_jobs_df = final_dims_jobs_df[dim_jobs_columns]
facts_df = facts_df[facts_jobs_columns]
dim_location_df = dim_location_df[dim_location_columns]
dma_df = dma_df[dim_dma_columns]

sql_file_path = "dim_modeling.sql"

create_bigquery_schema(sql_file_path=sql_file_path)

## ingest tables into schema

insert_dataframe_to_bigquery(df=final_dims_jobs_df,
                             dataset_table_name='living_wages_project.dim_jobs',
                             project_id=PROJECT_ID,
                             if_exists='replace')
insert_dataframe_to_bigquery(df=dim_dma_df,
                             dataset_table_name='living_wages_project.dim_dma',
                             project_id=PROJECT_ID,
                             if_exists='replace')
insert_dataframe_to_bigquery(df=dim_location_df,
                             dataset_table_name='living_wages_project.dim_location',
                             project_id=PROJECT_ID,
                             if_exists='replace')
insert_dataframe_to_bigquery(df=facts_df,
                             dataset_table_name='living_wages_project.facts_jobs',
                             project_id=PROJECT_ID,
                             if_exists='replace')

facts_df.to_csv("facts_jobs.csv", index=False)
final_dims_jobs_df.to_csv("dim_jobs.csv", index=False)

print("ETL process is complete.")
