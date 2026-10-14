#Build an ML Pipeline for Short-Term Rental Prices in NYC

This project builds a machine learning pipeline that predicts the price of Airbnb rentals in New York City.
The pipeline uses MLflow to run each step and Weights & Biases (W&B) to track data, runs, and models.

##Project Links
W&B project: https://forge.coreweave.com/wandb/ccraw85-western-governors-university/nyc_airbnb/overview
GitHub repository: https://github.com/crawforc/Project-Build-an-ML-Pipeline-Starter

##Pipeline Steps
download - downloads a sample of the data and uploads it to W&B as `sample.csv`
basic_cleaning - removes price outliers and converts `last_review` to a date, then saves `clean_sample.csv`
data_check - runs tests to make sure the cleaned data looks correct
data_split - splits the data into a train/validation set and a test set
train_random_forest - trains a random forest model and saves it to W&B
test_regression_model - tests the production model on the test set

##How to Run
Create and activate the conda environment:
```
conda env create -f environment.yml
conda activate nyc_airbnb_dev
```
Run the whole pipeline:
```
mlflow run .
```
Run only some steps:
```
mlflow run . -P steps=download,basic_cleaning
```
Test the production model (this step must be run on its own):
```
mlflow run . -P steps=test_regression_model
```
##Results
I tried 4 combinations of hyperparameters. The best model used:
`max_depth`: 50
`n_estimators`: 200

Results on the test set:
MAE: 33.85
R2: 0.562

##Discussion
Model could be improved with a more comprehensive dataset, the data does not include parameters for apartment size, number of bedrooms, amenities, etc that would improve the accuracy of the model. 