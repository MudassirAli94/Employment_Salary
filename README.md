# Overview

Companies need to take into account different sets of variables when deciding on a salary range for their job listings. Amongst those variables are: location, type of job, number of years of experience a candidate has, average living expenses and etc. The following project takes into account the following sources:

- census
- MIT living wages
- levels FYI
- minimum wages per state
- DMA

creating dimensional modeling, breaking down these sources into a data warehouse via Google Cloud Platform (GCP) and producing a predictive model that performs an analysis on breaking down the top coefficents for predicting a 95% confidence interval on the salary range.

# Data Engineering

## Information Architecture:

![image](https://github.com/KaiwenLian/CIS9440Group9/assets/38592433/e379d71a-c931-4d4c-8308-4b6b027357a7)


## Dimensional Modeling:

![image](https://github.com/KaiwenLian/CIS9440Group9/assets/38592433/c97dfdf7-f8ab-421d-b13e-1a6a0b6054ae)

## Extract

Using various APIs , python packages and public data via webscrapping on the aformentioned data sources we were able to extract the data in python and land the data in the staging area of Google Cloud Storage. You can find the extract python script here:

https://github.com/KaiwenLian/CIS9440Group9/blob/main/ETL%20Scripts/extract_data.py

## Transform

Transforming the data required numerous steps as there was lots of cleaning and preprocessing needed to be done which can be found here:

https://github.com/KaiwenLian/CIS9440Group9/blob/main/ETL%20Scripts/transform_data.py

## Load

By using the package pandas_gbq we can seamlessly create our own functions to load data into the data warehouse, Google BigQuery. The script is also in the transform_data.py script above.

# Data Analysis

## Initial Steps

To perform a thorough analysis we decided to data mine the possible features affecting one's salary. In order to properly assess this problem there were several small models incorporated into the data to predict the null values living in our dataset. The following columns had null values in which we predicted the null values for our final model.

- Salary
- Years of experience
- Years at level
- MIT Living Wages

Rather than dropping null values or filling them with the mean/median which can skew the results, we ran several smaller models to predict, providing us a more enriching understanding of the data.

## Modeling - Catboost Regressor

The type of model used is a Catboost Regressor (regressor since we are dealing with numerical values). 

CatBoost, short for Categorical Boosting, is a specialized boosting model designed to efficiently handle categorical data. Unlike other machine learning algorithms, CatBoost does not require preprocessing techniques like dummy or one-hot encoding, which can increase the dataset's complexity and size. This simplifies the model training process and often leads to better performance. Furthermore, CatBoost implements an ordered boosting algorithm, which uses random permutations of the dataset to train sequential models. This approach helps to avoid data leakage and ensures that each model learns from a truly independent subset of the data, enhancing the overall robustness and accuracy of the predictions.

More on catboosting here: https://catboost.ai/

## Model Results

| Model | Mean Absolute Error (MAE) | R Squared |
|----------|----------|----------|
| Null Value - Salary    | 32395    | 0.40     |
| Null Value - Years of Exp.   | 3.03    | 0.47    |
| Null Value - Years at Level.   | 0.96    | 0.17     |
| Null Value - MIT Living Wages   | 1800    | 0.98    |
| Final Model - Salary   | 24374     | 0.66    |


