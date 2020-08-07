import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

import warnings
warnings.filterwarnings('ignore')

import datetime
import glob
import os
from pathlib import Path

from cloud_storage import *

# Retreive correct data files 
# from postgres_upload import to_predictions_db
from utils_config import Config

CURR_SEASON = Config.get("curr_season")
yesterday = Config.get("YESTERDAY")
today = Config.get("TODAY")

output_path = Path(f"data/predictions/season_{CURR_SEASON}/")
output_path.mkdir(parents=True, exist_ok=True)

def get_data():
  data_path = Path(f"data/player_stats/season_{CURR_SEASON}")
  data_path.mkdir(parents=True, exist_ok=True)
  get_prev_k_blobs(data_path, k=8)
  data_files = glob.glob(f"data/player_stats/season_{CURR_SEASON}/*.csv")
  data_files.sort()
  last_7_generated = data_files[-8:-1]

  print("Getting data...")

  hits = pd.concat([pd.read_csv(f) for f in last_7_generated], sort=False)
  print(hits.iloc[0][:10])
  print(f"Total rows: {len(hits)}")
  hits.dropna(inplace=True)
  hits.drop_duplicates(inplace=True)
  hits['pitcher_hitter_opposite_hand'] = hits['pitcher_hitter_opposite_hand'].astype(float)
  print(f"Total rows after dropping missing values: {len(hits)}")
  hits.set_index(np.arange(len(hits)), inplace=True)
  hits['player_got_hit'] = hits['player_got_hit'].apply(float)

  label_encoder = LabelEncoder()
  data = hits.drop(['date', 'Name', 'Team', 'ID', 'player_got_hit'], axis='columns')
  labels = label_encoder.fit_transform(hits['player_got_hit'])

  hits_test = pd.read_csv(data_files[-1])
  hits_test.dropna(inplace=True)
  hits_test.drop_duplicates(inplace=True)
  hits_test['pitcher_hitter_opposite_hand'] = hits_test['pitcher_hitter_opposite_hand'].astype(float)
  data_test = hits_test.drop(['date', 'Name', 'Team', 'ID'], axis='columns')

  print("Data retrieved.")
  print(f"Length of data: {len(data)}")
  print(f"Length of labels: {len(labels)}")
  print(f"Length of test data: {len(data_test)}")

  return data, labels, hits_test, data_test

train_data, train_labels, hits_test, data_test = get_data()
data_train, data_val, labels_train, labels_val = train_test_split(train_data, train_labels, test_size=0.1)

# LOGISTIC REGRESSION

# Default params
print("Training logistic regression...")
logreg = LogisticRegression(penalty='l2').fit(train_data, train_labels)
print("Finished training!")

# XGBoost


dtrain = xgb.DMatrix(data_train, label=labels_train) # This works because XGBoost works with Pandas
dval = xgb.DMatrix(data_val, label=labels_val)
dtest = xgb.DMatrix(data_test)

xgb_params = {'objective': 'binary:logistic'}
xgb_grid_search = {
  'max_depth': [5, 10, 20, 40],
  'min_child_weight': [2, 5, 10, 15]
}

def xgboost_cv(xgb_params, xgb_grid_search, dtrain, n_fold=5):
  print("Finding best hyperparameters for XGBoost...")
  largest_auroc = float("-inf")
  best_params = {'max_depth': None, 'min_child_weight': None}
  for md in xgb_grid_search['max_depth']:
    for mcw in xgb_grid_search['min_child_weight']:

      curr_params = {"max_depth": md, "min_child_weight": mcw}
      xgb_params.update(curr_params)

      xgb_cv_results = xgb.cv(
            xgb_params,
            dtrain,
            nfold=n_fold,
            metrics={'auc'}
        )

      mean_auroc = xgb_cv_results['test-auc-mean'].min()
      if mean_auroc > largest_auroc:
        best_params.update(curr_params)

  return best_params

xgb_params.update(xgboost_cv(xgb_params, xgb_grid_search, dtrain))
print("Training XGBoost...")
boosted_dt = xgb.train(xgb_params, dtrain)
print("Finished training!")

# RANDOM FORESTS

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

def random_forest_cv(train_data, train_labels):
  print("Finding best hyperparameters for random forests...")
  param_grid = [
      {'criterion': ['gini'], 'max_depth': [20], 'min_samples_leaf': [4, 10, 20, 30],
      'n_estimators': [50, 75, 100, 125, 150]}
  ]

  rf_cv = GridSearchCV(RandomForestClassifier(), param_grid, cv=4)
  rf_cv.fit(train_data, train_labels)
  best_rf_params = rf_cv.best_params_

  return best_rf_params

print("Training random forests...")
rf_classifier = RandomForestClassifier()
rf_classifier.set_params(**random_forest_cv(data_train, labels_train))
rf_classifier.fit(data_train, labels_train)
print("Finished training!")

# Get model summaries
def get_overall_performance():
  acc_logreg = np.mean(logreg.predict(data_val) == labels_val)
  prec_logreg = precision_score(labels_val, logreg.predict(data_val))
  rec_logreg = recall_score(labels_val, logreg.predict(data_val))
  f1_logreg = f1_score(labels_val, logreg.predict(data_val))

  acc_rf = np.mean(rf_classifier.predict(data_val) == labels_val)
  prec_rf = precision_score(labels_val, rf_classifier.predict(data_val))
  rec_rf = recall_score(labels_val, rf_classifier.predict(data_val))
  f1_rf = f1_score(labels_val, rf_classifier.predict(data_val))

  xgb_predictions = (boosted_dt.predict(dval) >  0.5).astype(float)
  acc_xgb = np.mean(xgb_predictions == labels_val)
  prec_xgb = precision_score(labels_val, xgb_predictions)
  rec_xgb = recall_score(labels_val, xgb_predictions)
  f1_xgb = f1_score(labels_val, xgb_predictions)

  performance = pd.DataFrame([['Logreg', acc_logreg, prec_logreg, rec_logreg, f1_logreg],
               ['Random Forests', acc_rf, prec_rf, rec_rf, f1_rf],
               ['XGBoost', acc_xgb, prec_xgb, rec_xgb, f1_xgb]], 
               columns=['Model', 'Accuracy', 'Precision', 'Recall', "F1 Score"])
  print("Model performance summary on validation set: \n", performance)
  return performance

performance = get_overall_performance()
performance.to_csv("data/model_stats/performance_{}.csv".format(today.replace('/', '_')), index=False)

models_dict = {'Logreg': logreg, 'Random Forests': rf_classifier, 'XGBoost': boosted_dt}
best_f1_model = performance.sort_values('F1 Score', ascending=False).iloc[0]['Model']
best_model = models_dict[best_f1_model]

# Make predictions

hit_probs = None
if best_f1_model == "XGBoost":
  # Predictions format is slightly different for XGBoost
  xgbpreds = best_model.predict(dtest)
  hit_probs = xgbpreds
  predictions = hits_test.take(np.argsort(xgbpreds)[::-1][:10])[['Name', 'ID', 'Team']].reset_index().iloc[:, 1:]
else:
  test_preds = best_model.predict_proba(data_test)[:, 1]
  hit_probs = test_preds
  predictions = hits_test.take(np.argsort(test_preds)[::-1][:10])[['Name', 'ID', 'Team']].reset_index().iloc[:, 1:]
predictions["hit_probability"] = np.sort(hit_probs)[::-1][:10]
predictions['Date'] = today
predictions['prediction_id'] = predictions['Date'].str.cat(predictions['ID'].apply(lambda x: str(x)))
predictions.drop("ID", axis='columns', inplace=True)
predictions.columns = ['Name', 'Team', 'Hit Probability', 'Date', 'prediction_id']
file_to_generate = output_path / f"predictions_{today.replace('/', '_')}.csv"
predictions.to_csv(file_to_generate, index=False)

print("Predictions for today: \n", predictions)

upload_blob(source_file_name=str(file_to_generate), destination_blob_name=str(file_to_generate))
# Add data for accuracy plot visualization for website

# accuracy_plot_data = pd.read_csv("data/plots/accuracy_plot_data.csv")
# new_day = today[:-5] # "7/20", for example
# latest_overall_acc = max(performance['Accuracy'])
# latest_top10_acc = float(predictions.iloc[-1]['player_got_hit'])
# accuracy_plot_data = accuracy_plot_data.append({"Day": new_day, 
#                            "Overall Accuracy": latest_overall_acc, 
#                            "Top 10 Accuracy": latest_top10_acc}, 
#                           ignore_index=True)
# accuracy_plot_data.to_csv("data/plots/accuracy_plot_data.csv")
