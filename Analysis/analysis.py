import pandas as pd
import numpy as np
from gcp_functions import read_table_from_bq, insert_dataframe_to_bigquery
import json
from catboost import CatBoostRegressor
pd.set_option('display.max_columns', None)
import warnings
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import math
from statsmodels.stats.outliers_influence import variance_inflation_factor
import shap
from statsmodels.tools.tools import add_constant
import matplotlib.pyplot as plt
from tqdm import tqdm

warnings.filterwarnings("ignore")

with open('config.json') as config_file:
    config = json.load(config_file)

YOUR_BUCKET_NAME = config["bucket_name"]
PROJECT_ID = config["project_id"]

# Read the data from BigQuery into a pandas DataFrame

facts_query = f"""
SELECT * FROM living_wages_project.facts_jobs
"""

dim_job_query = f"""
SELECT * FROM living_wages_project.facts_jobs_salary
"""

facts_df = read_table_from_bq(facts_query, project_id=PROJECT_ID)
dim_job_df = read_table_from_bq(dim_job_query, project_id=PROJECT_ID)

df = facts_df.merge(dim_job_df[["job_title","job_id", "job_family","occupational_area", "company_name","salary"]], on="job_id").drop_duplicates(subset=["job_id"])
df = df[df["job_family"] != "Other"]

df["company_name"] = df["company_name"].fillna("Not Listed or Start Up")

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

cols = ["job_id","job_family", "company_name","occupational_area", "head_flag", "lead_flag", "senior_flag", "vp_flag", "avp_flag",
        "director_flag", "executive_flag","chief_flag", "president_flag", "total_population_density" , "location_id"]
df_salary_null = df_salary_null[cols]

df_salary_null = df_salary_null.dropna()
df_salary_null.set_index("job_id", inplace=True)

assert df_salary_null.shape[0] > 0 , "There are no values in the salary column"

print("length of null salary dataframe", df_salary_null.shape[0])


cols.append("salary")
df_salary = df[(df.salary > 0)]
#df_salary["salary"] = df_salary["salary"].apply(lambda i: np.log((i+1)/1000))
df_salary = df_salary[cols]
df_salary = df_salary.dropna()
df_salary.set_index("job_id", inplace=True)

assert df_salary.shape[0] > 0 , "There are no values in the salary column"

print("length of salary dataframe", df_salary.shape[0])

X = df_salary.drop('salary', axis=1)
y = df_salary['salary']

# Define categorical features
categorical_features = ['job_family', 'occupational_area', "location_id" ,"company_name"]

## train test split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor
model = CatBoostRegressor(cat_features=categorical_features, random_seed=42)

## fit train and test
model.fit(X_train, y_train, eval_set=(X_test,y_test) , verbose=False)

predictions = model.predict(X_test)

#predictions = np.exp(predictions) - 1

#y_test = y_test.apply(lambda i: np.exp(i)*1000)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print()
print("Mean Absolute Error (MAE) for predicting salary:", mae)

# Calculate R-squared
r2 = r2_score(y_test, predictions)

print("R-squared (R²) for predicting salary:", r2)
print()



model.fit(X,y,verbose=False)

# Make predictions
predictions = model.predict(df_salary_null)
#predictions = np.exp(predictions) - 1


df_salary_null["salary"] = predictions
df_salary_null["salary"] = df_salary_null["salary"].apply(lambda i: math.floor(i))

df_salary_full = pd.concat([df_salary.reset_index(), df_salary_null.reset_index()])[["job_id","salary"]]
df_not_in_pred = df[~df.job_id.isin(df_salary_full.job_id)]
df = df.drop(columns=["salary"]).merge(df_salary_full, on="job_id")

df = pd.concat([df, df_not_in_pred])

## now for years of experience
df_yrs_exp_null = df[df.years_of_experience.isnull()==True]
cols = ["job_id","job_family","occupational_area", "company_name" ,"head_flag", "lead_flag", "senior_flag", "vp_flag", "avp_flag",
        "director_flag", "executive_flag","chief_flag", "president_flag", "salary" , "location_id"]
df_yrs_exp_null = df_yrs_exp_null[cols]
df_yrs_exp_null = df_yrs_exp_null.dropna()
df_yrs_exp_null.set_index("job_id", inplace=True)

assert df_yrs_exp_null.shape[0] > 0 , "There are no values in the years of experience column"


cols.append("years_of_experience")
df_yrs = df[df.years_of_experience.isnull()==False]
#df_yrs["years_of_experience"] = df_yrs["years_of_experience"].apply(lambda i: np.log((i+1)/1000))
df_yrs = df_yrs[cols]
df_yrs = df_yrs.dropna()
df_yrs.set_index("job_id", inplace=True)

assert df_yrs.shape[0] > 0 , "There are no values in the years of experience column"

## run cat boost regression on years of experience

X = df_yrs.drop('years_of_experience', axis=1)
y = df_yrs['years_of_experience']

# Define categorical features
categorical_features = ['job_family', 'occupational_area', "location_id", "company_name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor
model = CatBoostRegressor(cat_features=categorical_features, random_seed=42)

# Fit the model

model.fit(X_train, y_train, eval_set=(X_test,y_test) , verbose=False)

predictions = model.predict(X_test)
#predictions = np.exp(predictions) - 1

#y_test = y_test.apply(lambda i: np.exp(i)*1000)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print()
print("Mean Absolute Error (MAE) for predicting years of experience:", mae)

# Calculate R-squared
r2 = r2_score(y_test, predictions)
print("R-squared (R²) for predicting years of experience:", r2)
print()

model.fit(X, y, verbose=False)

# Make predictions
predictions = model.predict(df_yrs_exp_null)
#predictions = np.exp(predictions) - 1

df_yrs_exp_null["years_of_experience"] = predictions
df_yrs_exp_null["years_of_experience"] = df_yrs_exp_null["years_of_experience"].apply(lambda i: math.floor(i))

df_yrs_full = pd.concat([df_yrs.reset_index(), df_yrs_exp_null.reset_index()])[["job_id","years_of_experience"]]

df_not_in_pred = df[~df.job_id.isin(df_yrs_full.job_id)]
df = df.drop(columns=["years_of_experience"]).merge(df_yrs_full, on="job_id")

df = pd.concat([df, df_not_in_pred])



## now use same exact steps to predict years_at_level

df_yrs_at_level_null = df[df.years_at_level.isnull()==True]
cols = ["job_id","job_family","occupational_area", "head_flag", "lead_flag", "senior_flag", "vp_flag", "avp_flag",
        "director_flag", "executive_flag","chief_flag", "president_flag", "salary", "years_of_experience", "location_id",
        "company_name"]
df_yrs_at_level_null = df_yrs_at_level_null[cols]
df_yrs_at_level_null = df_yrs_at_level_null.dropna()
df_yrs_at_level_null.set_index("job_id", inplace=True)

assert df_yrs_at_level_null.shape[0] > 0 , "There are no values in the years at level column"

cols.append("years_at_level")
df_yrs_at_level = df[df.years_at_level.isnull()==False]
#df_yrs_at_level["years_at_level"] = df_yrs_at_level["years_at_level"].apply(lambda i: np.log((i+1)/1000))
df_yrs_at_level = df_yrs_at_level[cols]
df_yrs_at_level = df_yrs_at_level.dropna()
df_yrs_at_level.set_index("job_id", inplace=True)

assert df_yrs_at_level.shape[0] > 0 , "There are no values in the years at level column"

## run cat boost regression on years of experience

X = df_yrs_at_level.drop('years_at_level', axis=1)
y = df_yrs_at_level['years_at_level']

# Define categorical features

categorical_features = ['job_family', 'occupational_area', "location_id", "company_name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor

model = CatBoostRegressor(cat_features=categorical_features, random_seed=42)

# Fit the model

model.fit(X_train, y_train, eval_set=(X_test,y_test) , verbose=False)

predictions = model.predict(X_test)
#predictions = np.exp(predictions-1)

#y_test = y_test.apply(lambda i: np.exp(i)*1000)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, predictions)
print()
print("Mean Absolute Error (MAE) for years at level:", mae)

# Calculate R-squared
r2 = r2_score(y_test, predictions)
print("R-squared (R²) for years at level:", r2)
print()

model.fit(X, y, verbose=False)

# Make predictions

predictions = model.predict(df_yrs_at_level_null)
#predictions = np.exp(predictions) - 1

df_yrs_at_level_null["years_at_level"] = predictions
df_yrs_at_level_null["years_at_level"] = df_yrs_at_level_null["years_at_level"].apply(lambda i: math.floor(i))

df_yrs_at_level_full = pd.concat([df_yrs_at_level.reset_index(), df_yrs_at_level_null.reset_index()])[["job_id","years_at_level"]]
df_not_in_pred = df[~df.job_id.isin(df_yrs_at_level_full.job_id)]

df = df.drop(columns=["years_at_level"]).merge(df_yrs_at_level_full, on="job_id")

df = pd.concat([df, df_not_in_pred])

print(df.info())

## now run catboost on mit_estimated_baseline_salary

df_mit_null = df[df.mit_estimated_baseline_salary.isnull()==True]
cols = ["job_id","job_family","occupational_area", "head_flag", "lead_flag", "senior_flag", "vp_flag", "avp_flag",
        "director_flag", "executive_flag","chief_flag", "president_flag", "salary", "years_of_experience",
        "years_at_level", "location_id", "company_name"]
df_mit_null = df_mit_null[cols]
df_mit_null = df_mit_null.dropna()
df_mit_null.set_index("job_id", inplace=True)

assert df_mit_null.shape[0] > 0 , "There are no values in the mit_estimated_baseline_salary column"

cols.append("mit_estimated_baseline_salary")
df_mit = df[df.mit_estimated_baseline_salary.isnull()==False]
#df_mit["mit_estimated_baseline_salary"] = df_mit["mit_estimated_baseline_salary"].apply(lambda i: np.log((i+1)/1000))
df_mit = df_mit[cols]
df_mit = df_mit.dropna()
df_mit.set_index("job_id", inplace=True)

assert df_mit.shape[0] > 0 , "There are no values in the mit_estimated_baseline_salary column"
print(df_mit.shape)

X = df_mit.drop('mit_estimated_baseline_salary', axis=1)
y = df_mit['mit_estimated_baseline_salary']

# Define categorical features

categorical_features = ['job_family', 'occupational_area', "location_id", "company_name"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor

model = CatBoostRegressor(cat_features=categorical_features, random_seed=42, l2_leaf_reg=3)

# Fit the model

model.fit(X_train, y_train, eval_set=(X_test,y_test) , verbose=False)

predictions = model.predict(X_test)

#predictions = np.exp(predictions) - 1

#y_test = y_test.apply(lambda i: np.exp(i)*1000)

# Calculate Mean Absolute Error

mae = mean_absolute_error(y_test, predictions)

print()
print("Mean Absolute Error (MAE) for predicting mit_estimated_baseline_salary:", mae)
print()

r2 = r2_score(y_test, predictions)
print("R-squared (R²) for predicting mit_estimated_baseline_salary:", r2)

model.fit(X, y, verbose=False)

# Make predictions

predictions = model.predict(df_mit_null)
#predictions = np.exp(predictions) - 1

df_mit_null["mit_estimated_baseline_salary"] = predictions
df_mit_null["mit_estimated_baseline_salary"] = df_mit_null["mit_estimated_baseline_salary"].apply(lambda i: math.floor(i))

df_mit_full = pd.concat([df_mit.reset_index(), df_mit_null.reset_index()])[["job_id","mit_estimated_baseline_salary"]]
df_not_in_pred = df[~df.job_id.isin(df_mit_full.job_id)]

df = df.drop(columns=["mit_estimated_baseline_salary"]).merge(df_mit_full, on="job_id")

df = pd.concat([df, df_not_in_pred])

## need to take average of mit_estimated_baseline_salary for each location_id

df_mit_estimated = df.groupby("location_id")["mit_estimated_baseline_salary"].mean().reset_index()
df = df.drop(columns=["mit_estimated_baseline_salary"]).merge(df_mit_estimated, on="location_id")

print(df[["mit_estimated_baseline_salary", "location_id"]].head(10))

print(df.info())



## now run catboost regressor on full salary to get relevant coeficients

cols = ["job_id","job_family","occupational_area", "total_population_density", "total_land_area",
        "years_of_experience","years_at_level", "salary", "total_housing_units",
        "location_id", "mit_estimated_baseline_salary", "company_name"]

cols_to_check_multi_collinearity = ["total_population_density", "total_land_area", "total_housing_units"]

vif_df = df[cols_to_check_multi_collinearity].dropna()

vif_df = add_constant(vif_df)
vif_data = pd.DataFrame()
vif_data["feature"] = vif_df.columns
vif_data["VIF"] = [variance_inflation_factor(vif_df.values, i) for i in range(vif_df.shape[1])]

print(vif_data)
print()

multicollinarity_cols = list(vif_data[vif_data["VIF"] > 10]["feature"])

# remove cols with < 10 VIF

cols = [col for col in cols if col not in multicollinarity_cols]

df = df[cols]
df = df.dropna()
df = df.drop_duplicates(subset=["job_id"])
#df["salary"] = df["salary"].apply(lambda i: np.log((i+1)/1000))
df.set_index("job_id", inplace=True)

assert df.shape[0] > 0 , "There are no values in the salary column"

X = df.drop('salary', axis=1)
y = df['salary']

# Define categorical features

categorical_features = ['job_family', 'occupational_area', "location_id", "company_name"]

## train test split to get r squared value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize CatBoostRegressor

model = CatBoostRegressor(cat_features=categorical_features, random_seed=42)

# Fit the model

model.fit(X_train, y_train, eval_set=(X_test,y_test) , verbose=False)

predictions = model.predict(X_test)

#predictions = np.exp(predictions)-1 * 1000

#y_test = y_test.apply(lambda i: np.exp(i)-1 * 1000)

# Calculate Mean Absolute Error

mae = mean_absolute_error(y_test, predictions)
print()
print("Mean Absolute Error (MAE) for predicting salary:", mae)
print()

r2 = r2_score(y_test, predictions)
print("R-squared (R²) for predicting salary:", r2)
print()

## run model on whole dataset to get feature importances

model = CatBoostRegressor(cat_features=categorical_features, random_seed=42)

# Fit the model on the entire dataset
model.fit(X, y, verbose=False)

## save model for future use

# data = {
#
#     "job_id" : ["101"],
#     "job_family" : ["Computer & Mathematical"],
#     "occupational_area" : ["Computer & Mathematical"],
#     "total_population_density": [5670],
#     "total_land_area": [5.2],
#     "years_of_experience": [10],
#     "years_at_level" : [5],
#     "total_housing_units": [11872],
#     "location_id" : ["Middletown_New York"]
# }
#
# data_df = pd.DataFrame(data)
#
# data_df = data_df.set_index('job_id')
#
# print(model.predict(data_df))
#
# model.save_model('catboost_model.cbm', format="cbm")
#
# print(X.head())
#
# import sys
# sys.exit(1)

#Bootstrap parameters
n_bootstraps = 50
alpha = 0.05
lower_quantile = alpha/2
upper_quantile = 1 - (alpha/2)

# Store predictions from each bootstrap
bootstrap_predictions = []

for _ in tqdm(range(n_bootstraps), desc="Calculating Bootstraps"):
    # Sample with replacement from X and y
    indices = np.random.choice(range(X.shape[0]), size=X.shape[0], replace=True)
    X_sample = X.iloc[indices]
    y_sample = y.iloc[indices]

    # Fit the model on the bootstrap sample
    model.fit(X_sample, y_sample, verbose=False)

    # Predict on the original dataset
    predictions = model.predict(X)
    bootstrap_predictions.append(predictions)

# Calculate quantiles for the confidence interval
bootstrap_predictions = np.array(bootstrap_predictions)
lower_bounds = np.percentile(bootstrap_predictions, lower_quantile * 100, axis=0)
upper_bounds = np.percentile(bootstrap_predictions, upper_quantile * 100, axis=0)

# Create DataFrame
X_predicted_df = X.reset_index()
X_predicted_df["predicted_salary"] = model.predict(X)
X_predicted_df["predicted_salary"] = X_predicted_df["predicted_salary"].apply(np.floor)
X_predicted_df["confidence_interval_lower_bound"] = lower_bounds
X_predicted_df["confidence_interval_upper_bound"] = upper_bounds

new_facts_df = read_table_from_bq(facts_query, project_id=PROJECT_ID)

new_facts_df = new_facts_df.merge(X_predicted_df[["job_id", "predicted_salary",
                                      "confidence_interval_lower_bound","confidence_interval_upper_bound"]], on="job_id")

new_facts_df.drop_duplicates(subset=["job_id"], inplace=True)

insert_dataframe_to_bigquery(df=new_facts_df,
                             dataset_table_name='living_wages_project.facts_jobs_salary',
                             project_id=PROJECT_ID,
                             if_exists='replace')


feature_importances = model.get_feature_importance()

feature_names = X.columns

for score, name in sorted(zip(feature_importances, feature_names), reverse=True):
    print(f"{name} coefficient: {np.round(score,3)}")




explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

plt.ioff()
plt.switch_backend('agg')

print(shap.summary_plot(shap_values, X, plot_type="bar"))
plt.savefig("shap_summary_plot.png")
plt.clf()


lower_range = 0
upper_range = 50000
shap_values = explainer(X.iloc[lower_range:upper_range])

for n in tqdm(range(upper_range - lower_range), desc="Generating SHAP plots"):
    plt.figure(figsize=(12, 6))
    shap.plots.waterfall(shap_values[n], show=False)
    plt.savefig(
        fr"C:\Users\Mudas\Documents\school\Baruch\Data Warehouse\data\Term Project\Analysis\shap_plots\shap_waterfall_plot_{n}.png",
        dpi=300)

    # Clear the current figure to prepare for the next plot
    plt.clf()



print()
print("finished running script")


