import pandas as pd
import numpy as np
from gcp_functions import read_table_from_bq
import json
from catboost import CatBoostRegressor
pd.set_option('display.max_columns', None)
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import math

with open('config.json') as config_file:
    config = json.load(config_file)

YOUR_BUCKET_NAME = config["bucket_name"]
PROJECT_ID = config["project_id"]

# Read the data from BigQuery into a pandas DataFrame

facts_query = f"""
SELECT * FROM living_wages_project.facts_jobs
"""

dim_job_query = f"""
SELECT * FROM living_wages_project.dim_jobs
"""

facts_df = read_table_from_bq(facts_query, project_id=PROJECT_ID)
dim_job_df = read_table_from_bq(dim_job_query, project_id=PROJECT_ID)

df = facts_df.merge(dim_job_df[["job_title","job_id", "job_family","occupational_area"]], on="job_id").drop_duplicates(subset=["job_id"])

## make flagged columns if the job is: head, lead, senior, vp, avp, director

df["job_title"] = df.job_title.str.lower()
df["job_title"] = df.job_title.str.replace("vice president","vp")
df["job_title"] = df.job_title.str.replace("assistant vice president","avp")

df["chief_flag"] = df.job_title.str.contains("chief", case=False).astype(int)
df["head_flag"] = df.job_title.str.contains("head", case=False).astype(int)
df["lead_flag"] = df.job_title.str.contains("lead", case=False).astype(int)
df["senior_flag"] = df.job_title.str.contains("senior", case=False).astype(int)
df["vp_flag"] = df.job_title.str.contains("vp", case=False).astype(int)
df["president_flag"] = df.job_title.str.contains("president", case=False).astype(int)
df["avp_flag"] = df.job_title.str.contains("avp", case=False).astype(int)
df["director_flag"] = df.job_title.str.contains("director", case=False).astype(int)
df["executive_flag"] = df.job_title.str.contains("director", case=False).astype(int)

df = df.drop(columns = ["job_title"])

## begin ML on missing rows of columns

## run cat boost regression on salary

df_salary_null = df[(df.salary == 0)]
# cols = list(df_salary_null.drop(columns=["salary", "dma_id","location_id", "years_of_experience" , "mit_estimated_baseline_salary",
#                                          "years_at_level"]).columns)

cols = ["job_id","job_family","occupational_area", "head_flag", "lead_flag", "senior_flag", "vp_flag", "avp_flag",
        "director_flag", "executive_flag","chief_flag", "president_flag"]
df_salary_null = df_salary_null[cols]

df_salary_null = df_salary_null.dropna()
df_salary_null.set_index("job_id", inplace=True)

assert df_salary_null.shape[0] > 0 , "There are no values in the salary column"

print("length of null salary dataframe", df_salary_null.shape[0])


cols.append("salary")
df_salary = df[(df.salary > 0)]
df_salary["salary"] = df_salary["salary"].apply(lambda i: np.log(i+1))
df_salary = df_salary[cols]
df_salary = df_salary.dropna()
df_salary.set_index("job_id", inplace=True)

assert df_salary.shape[0] > 0 , "There are no values in the salary column"

print("length of salary dataframe", df_salary.shape[0])

X = df_salary.drop('salary', axis=1)
y = df_salary['salary']

# Define categorical features
categorical_features = ['job_family', 'occupational_area']

# Initialize CatBoostRegressor
model = CatBoostRegressor(cat_features=categorical_features, random_seed=42)


model.fit(X,y,verbose=False)

# Make predictions
predictions = model.predict(df_salary_null)
predictions = np.exp(predictions-1)


df_salary_null["salary"] = predictions
df_salary_null["salary"] = df_salary_null["salary"].apply(lambda i: math.floor(i))

df_salary_full = pd.concat([df_salary.reset_index(), df_salary_null.reset_index()])[["job_id","salary"]]
df_not_in_pred = df[~df.job_id.isin(df_salary_full.job_id)]
df = df.drop(columns=["salary"]).merge(df_salary_full, on="job_id")

df = pd.concat([df, df_not_in_pred])

## now for years of experience
df_yrs_exp_null = df[df.years_of_experience.isnull()==True]
cols = ["job_id","job_family","occupational_area", "head_flag", "lead_flag", "senior_flag", "vp_flag", "avp_flag",
        "director_flag", "executive_flag","chief_flag", "president_flag", "salary"]
df_yrs_exp_null = df_yrs_exp_null[cols]
df_yrs_exp_null = df_yrs_exp_null.dropna()
df_yrs_exp_null.set_index("job_id", inplace=True)

assert df_yrs_exp_null.shape[0] > 0 , "There are no values in the years of experience column"


cols.append("years_of_experience")
df_yrs = df[df.years_of_experience.isnull()==False]
df_yrs["years_of_experience"] = df_yrs["years_of_experience"].apply(lambda i: np.log(i+1))
df_yrs = df_yrs[cols]
df_yrs = df_yrs.dropna()
df_yrs.set_index("job_id", inplace=True)

assert df_yrs.shape[0] > 0 , "There are no values in the years of experience column"

## run cat boost regression on years of experience

X = df_yrs.drop('years_of_experience', axis=1)
y = df_yrs['years_of_experience']

# Define categorical features
categorical_features = ['job_family', 'occupational_area']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=df_yrs[categorical_features])

# Initialize CatBoostRegressor
model = CatBoostRegressor(cat_features=categorical_features, random_seed=42)

# Fit the model
model.fit(X_train, y_train, eval_set=(X_test, y_test), verbose=False)

# Make predictions
predictions = model.predict(X_test)
predictions = np.exp(predictions-1)

y_test = y_test.apply(lambda i: np.exp(i-1))

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print("Mean Absolute Error (MAE):", mae)

# Calculate R-squared
r2 = r2_score(y_test, predictions)
print("R-squared (R²):", r2)

df_yrs_exp_null["years_of_experience"] = predictions
df_yrs_exp_null["years_of_experience"] = df_yrs_exp_null["years_of_experience"].apply(lambda i: math.floor(i))

df_yrs_full = pd.concat([df_yrs.reset_index(), df_yrs_exp_null.reset_index()])[["job_id","years_of_experience"]]

df_not_in_pred = df[~df.job_id.isin(df_yrs_full.job_id)]
df = df.drop(columns=["years_of_experience"]).merge(df_yrs_full, on="job_id")

df = pd.concat([df, df_not_in_pred])


print(df.info())



## now use same exact steps to predict the 0 values of salary












print()
print("finished running script")

## now replace the null values of original dataframe with the predicted values by using the job_id


