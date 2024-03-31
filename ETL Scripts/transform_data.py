import numpy as np
import pandas as pd
import sys

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

levels_df = pd.read_csv("../levels_fyi.csv")

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
print(missing_percentages)
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

print(snake_case_columns)

## Rearrange columns

levels_df = levels_df[['uuid','state','state_short', 'city', 'title', 'job_family', 'level', 'focus_tag',
       'years_of_experience', 'years_at_company', 'years_at_level',
       'offer_date', 'work_arrangement', 'dma_id', 'base_salary', 'company_info_icon', 'company_info_name']]

## convert floats to int

for n in ["base_salary","dma_id"]:
    levels_df[n] = levels_df[n].astype(int)


print(levels_df.head())

## clean and transform mit living wages data

living_wage_df = pd.read_csv("../mit_living_wages.csv")

living_wage_df.typicalAnnualSalary = living_wage_df.typicalAnnualSalary.apply(lambda i: str(i).replace("$","")).apply(lambda i: str(i).replace(",",""))
living_wage_df.typicalAnnualSalary = living_wage_df.typicalAnnualSalary.astype(int)

cols_list = ["occupational_area","annual_salary","county_or_city","state"]
living_wage_df.columns = cols_list

living_wage_df["county_or_city"] = living_wage_df["county_or_city"].apply(lambda i: str(i).split(",")[0])

living_wage_df['state_short'] = living_wage_df['state'].map(state_abbreviations)
living_wage_df["county_or_city"] = living_wage_df["county_or_city"].apply(lambda i: i.replace(" County",""))
living_wage_df["county_or_city"] = living_wage_df["county_or_city"].apply(lambda i: i.replace(" city",""))
living_wage_df["county_or_city"] = living_wage_df["county_or_city"].apply(lambda i: i.replace(" Borough",""))
living_wage_df["county_or_city"] = living_wage_df["county_or_city"].apply(lambda i: i.split("-")[0])

print(living_wage_df.head())


## clean and transform minimum wage data

minimum_wage_df = pd.read_csv("../minimum_wage_per_state.csv")

cols_list = minimum_wage_df.columns.tolist()
cols_list = [col.replace(" ", "_").lower() for col in cols_list]
minimum_wage_df.columns = cols_list

for n in ["minimum_wage", "tipped_wage"]:

    minimum_wage_df[n] = minimum_wage_df[n].apply(lambda i: i.replace("$",""))
    minimum_wage_df[n] = minimum_wage_df[n].astype(float)


minimum_wage_df['state_short'] = minimum_wage_df['state'].map(state_abbreviations)

minimum_wage_df = minimum_wage_df[["state", "state_short", "minimum_wage", "tipped_wage"]]

print(minimum_wage_df.head())


## clean and transform start up jobs data
start_ups_df = pd.read_csv("../startups.csv")


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

print(start_ups_df.head())
## transform census data

import re

census_df = pd.read_csv("2020_census_data.csv")


census_df["county"] = census_df["NAME"].apply(lambda i: i.split(",")[0])
census_df["county"] = census_df["county"].apply(lambda i:str(i).replace('"',''))
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" County",""))
census_df["state"] = census_df["NAME"].apply(lambda i: i.split(", ")[1])
census_df["total_population"] = census_df["P1_001N"].astype(int)
census_df = census_df[["county", "state", "total_population"]]
census_df["state_short"] = census_df["state"].map(state_abbreviations)
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" Municipio",""))
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" Municipality",""))
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" City and Borough",""))
census_df["county"] = census_df["county"].apply(lambda i: i.replace(" city",""))

census_df = census_df[['state','state_short','county','total_population']]

print(census_df.head())

## transform gazetter data

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

gaz_df = gaz_df[['state','state_short','county','name','type_of_place','geo_id','ansi_code']]

print(gaz_df.head())
## transform DMA data

dma_df = pd.read_csv("dma.csv")



dma_df = dma_df.rename(columns = {"Designated Market Area (DMA)":"name","Rank":"rank","TV Homes":"tv_homes","% of US":"percent_of_united_states","DMA Code":"dma_id"})

dma_df.name = dma_df.name.apply(lambda i: i.split(",")[0])
dma_df.name = dma_df.name.apply(lambda i: i.split(" (")[0])
dma_df.name = dma_df.name.apply(lambda i: i.split("(")[0])
dma_df.name = dma_df.name.apply(lambda i: i.split("-")[0])
dma_df.name = dma_df.name.apply(lambda i: i.split(" &")[0])
dma_df.name = dma_df.name.apply(lambda i: i.split("&")[0])
dma_df.name = dma_df.name.apply(lambda i: i.replace(" City",""))
dma_df.name = dma_df.name.apply(lambda i: i.replace("Ft.","Fort"))
dma_df.name = dma_df.name.apply(lambda i: i.replace("Sacramnto","Sacramento"))
dma_df.name = dma_df.name.apply(lambda i: i.replace("SantaBarbra","Santa Barbara"))
dma_df.name = dma_df.name.apply(lambda i: i.replace("Idaho Fals","Idaho Falls"))
dma_df.name = dma_df.name.apply(lambda i: i.replace("Honolulu","Honolulu"))
dma_df.name = dma_df.name.apply(lambda i: i.replace("Rochestr","Rochester"))
dma_df.name = dma_df.name.apply(lambda i: i.replace("Greenvll","Greenville"))
dma_df.name = dma_df.name.apply(lambda i: i.replace("Wilkes Barre","Wilkes-Barre"))
dma_df.name = dma_df.name.apply(lambda i: i.replace("St Joseph","St. Joseph"))

dma_df = dma_df.sort_values(by="rank")

print(dma_df.head())

## transform efinancial data

# Define a function to remove string values from salary
efinancial_df=pd.read_csv("efinancial.csv")

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

print(df_job_data)
