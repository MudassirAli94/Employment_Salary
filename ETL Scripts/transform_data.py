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
    "District of Columbia": "DC", "Puerto Rico": "PR"
}

levels_df = pd.read_csv("../levels_fyi.csv")

levels_df['offerDate'] = levels_df['offerDate'].str.slice(start=0, stop=24)
# Now, convert the 'offerDate' column to datetime without specifying a format
levels_df['offerDate'] = pd.to_datetime(levels_df['offerDate'])
# Format the datetime as 'YYYYMMDD' string and then convert to integer
levels_df['offerDate'] = levels_df['offerDate'].dt.strftime('%Y%m%d').astype(int)

## getting short hand for state

levels_df["state_short"] = levels_df.location.apply(lambda i: str(i).split(', ')[-1])

## only interested in levels in USA

levels_df = levels_df[levels_df.state_short.isnull() != True]
levels_df = levels_df[levels_df.state_short.isin(state_abbreviations.values())]

## replace the brackets and quotes in the tags column

levels_df["tags"] = levels_df["tags"].apply(lambda i: str(i).replace("[",""))\
.apply(lambda i: str(i).replace("]","")).apply(lambda i: str(i).replace("'",""))\
    .apply(lambda i: str(i).replace(",","")).apply(lambda i: str(i).replace("-",""))\
    .apply(lambda i: np.nan if i == 'nan' else i)

## drop columns with all null values

for n in levels_df.columns:

    if levels_df[n].isnull().sum() == len(levels_df):
        levels_df.drop(columns = n, inplace = True)




## clean and transform mit living wages data

living_wage_df = pd.read_csv("../mit_living_wages.csv")

living_wage_df.typicalAnnualSalary = living_wage_df.typicalAnnualSalary.apply(lambda i: str(i).replace("$","")).apply(lambda i: str(i).replace(",",""))
living_wage_df.typicalAnnualSalary = living_wage_df.typicalAnnualSalary.astype(int)

cols_list = ["occupational_area","annual_salary","county","state"]
living_wage_df.columns = cols_list

living_wage_df.county = living_wage_df.county.apply(lambda i: str(i).split(",")[0])

living_wage_df['state_short'] = living_wage_df['state'].map(state_abbreviations)

print(living_wage_df.head())

print()



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


print(start_ups_df.num_employees.value_counts())

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
        # Add more mappings as necessary
    }
    # Handle generic 'Remote' cases or specific 'Remote US' cases by returning None or a specific value
    if 'Remote' in location or 'UAE' in location or 'Estonia' in location or 'Remote Switzerland' in location:
        return None

    # Map city names to states, and then states to abbreviations
    if location in city_to_state:
        state_name = city_to_state[location]
        return state_abbreviations.get(state_name, None)

    # Directly map known state names to abbreviations
    return state_abbreviations.get(location, None)


# Example usage:
start_ups_df['state_short'] = start_ups_df['location'].apply(lambda x: map_location_to_state(x, state_abbreviations))

# Drop rows with None in 'location' if needed

start_ups_df = start_ups_df[start_ups_df["state_short"].isnull() != True]

start_ups_df["location"] = start_ups_df["location"].replace(["San Francisco Bay Area","New York"],["San Francisco","New York City"])

start_ups_df = start_ups_df.rename(columns = {"location":"city"})

start_ups_df = start_ups_df[["job_title","city","state_short","salary","num_employees","date"]]

print(start_ups_df.head())
