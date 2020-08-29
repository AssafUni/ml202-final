import os
import time
import numpy as np
import pandas as pd
import pickle
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from logitboost import LogitBoost
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_score
from sklearn.utils.multiclass import type_of_target

class MadaBoostClassifier(AdaBoostClassifier):
  def _boost(self, iboost, X, y, sample_weight, random_state):
    n_samples = X.shape[0]
    d0 = 1 / n_samples
    new_weights = np.zeros(n_samples)
    sample_weight, estimator_weight, estimator_error = super()._boost(iboost, X, y, sample_weight, random_state)

    if sample_weight is None:
      return sample_weight, estimator_weight, estimator_error

    for idx, weight in enumerate(sample_weight):
      if weight < d0:
        new_weights[idx] = weight
      else:
        new_weights[idx] = d0
    
    return new_weights, estimator_weight, estimator_error

def remove_empty_lines(filename):
    if not os.path.isfile(filename):
        print('{} does not exist '.format(filename))
        return
    with open(filename) as filehandle:
        lines = filehandle.readlines()

    with open(filename, 'w') as filehandle:
        lines = filter(lambda x: x.strip(), lines)
        filehandle.writelines(lines)  

def fill_empty(X):
  for col in X.columns:
    if str(X[col].dtypes) == 'object':
      continue
    X[col] = X[col].fillna(0.0)

  return X

def get_dataset_dataframe(path_name):
  df = pd.read_csv(path_name)
  X = df.iloc[:, :-1]
  y = df.iloc[:, -1]
  X = fill_empty(X)

  return X, y

def get_all_files_paths(path_name):
  for dirname, _, filenames in os.walk(path_name):
      for filename in filenames:
          yield os.path.join(dirname, filename)

def get_all_datasets(path_name):
  for path in get_all_files_paths(path_name):
    try:
      X, y = get_dataset_dataframe(path)
      yield path, X, y
    except Exception as e:
      print('************** Failed to read {0}...'.format(path))
      print(e)
      raise e

def to_dict(X):
    return X.to_dict("records")

def get_scorer(target_type):
  if target_type == 'multiclass':
    return 'accuracy'
  elif target_type == 'binary':
    return 'roc_auc'
  else:
    raise Exception('Invalid target type')

def get_new_clf(clf_name, target_type, random_state):
  if clf_name == 'MadaBoost':
    dict_transformer = FunctionTransformer(to_dict, validate=False)
    pipeline = Pipeline([
        ('dict_transformer', dict_transformer),
        ('vectorizer', DictVectorizer(sparse=False)),
        ('clf', MadaBoostClassifier())
    ])
    param_grid = {
        'clf__base_estimator': [
          DecisionTreeClassifier(max_depth=1, random_state=random_state),
          DecisionTreeClassifier(max_depth=2, random_state=random_state),
          DecisionTreeClassifier(max_depth=8, random_state=random_state),
          DecisionTreeClassifier(max_depth=15, random_state=random_state),
          DecisionTreeClassifier(max_depth=22, random_state=random_state),
          DecisionTreeClassifier(max_depth=28, random_state=random_state),
          DecisionTreeClassifier(max_depth=36, random_state=random_state),
          DecisionTreeClassifier(max_depth=42, random_state=random_state),
          DecisionTreeClassifier(max_depth=48, random_state=random_state),           
        ],
        'clf__learning_rate': np.arange(0.05, 1, 0.05),
        'clf__n_estimators': np.arange(1, 1000, 50),
    }
    return RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_jobs=-1,
                              n_iter=50, scoring=get_scorer(target_type), verbose=1, cv=3, random_state=random_state)
  elif clf_name == 'AdaBoost':
    dict_transformer = FunctionTransformer(to_dict, validate=False)
    pipeline = Pipeline([
        ('dict_transformer', dict_transformer),
        ('vectorizer', DictVectorizer(sparse=False)),
        ('clf', AdaBoostClassifier())
    ])
    param_grid = {
        'clf__base_estimator': [
          DecisionTreeClassifier(max_depth=1, random_state=random_state),
          DecisionTreeClassifier(max_depth=2, random_state=random_state),
          DecisionTreeClassifier(max_depth=8, random_state=random_state),
          DecisionTreeClassifier(max_depth=15, random_state=random_state),
          DecisionTreeClassifier(max_depth=22, random_state=random_state),
          DecisionTreeClassifier(max_depth=28, random_state=random_state),
          DecisionTreeClassifier(max_depth=36, random_state=random_state),
          DecisionTreeClassifier(max_depth=42, random_state=random_state),
          DecisionTreeClassifier(max_depth=48, random_state=random_state),           
        ],
        'clf__learning_rate': np.arange(0.05, 1, 0.05),
        'clf__n_estimators': np.arange(1, 1000, 50),
    }
    return RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_jobs=-1,
                              n_iter=50, scoring=get_scorer(target_type), verbose=1, cv=3, random_state=random_state)
  elif clf_name == 'LogitBoost':
    dict_transformer = FunctionTransformer(to_dict, validate=False)
    pipeline = Pipeline([
        ('dict_transformer', dict_transformer),
        ('vectorizer', DictVectorizer(sparse=False)),
        ('clf', LogitBoost())
    ])
    param_grid = {
        'clf__base_estimator': [
          DecisionTreeRegressor(max_depth=1, random_state=random_state),
          DecisionTreeRegressor(max_depth=2, random_state=random_state),
          DecisionTreeRegressor(max_depth=8, random_state=random_state),
          DecisionTreeRegressor(max_depth=15, random_state=random_state),
          DecisionTreeRegressor(max_depth=22, random_state=random_state),
          DecisionTreeRegressor(max_depth=28, random_state=random_state),
          DecisionTreeRegressor(max_depth=36, random_state=random_state),
          DecisionTreeRegressor(max_depth=42, random_state=random_state),
          DecisionTreeRegressor(max_depth=48, random_state=random_state),           
        ],
        'clf__learning_rate': np.arange(0.05, 1, 0.05),
        'clf__n_estimators': np.arange(1, 1000, 50),
    }
    return RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_jobs=-1,
                              n_iter=50, scoring=get_scorer(target_type), verbose=1, cv=3, random_state=random_state) 
  elif clf_name == 'RandomForest':
    dict_transformer = FunctionTransformer(to_dict, validate=False)
    pipeline = Pipeline([
        ('dict_transformer', dict_transformer),
        ('vectorizer', DictVectorizer(sparse=False)),
        ('clf', RandomForestClassifier())
    ])
    param_grid = {
        'clf__min_samples_split': np.arange(2, 5, 50),
        'clf__n_estimators': np.arange(1, 1000, 50),
        'clf__random_state': [random_state],
    }
    return RandomizedSearchCV(estimator=pipeline, param_distributions=param_grid, n_jobs=-1,
                              n_iter=50, scoring=get_scorer(target_type), verbose=1, cv=3, random_state=random_state) 

def get_average_precision_score(y_true, y_true_multilabel_indicator, y_pred_prob, target_type):
  if target_type == 'multiclass':
    score_arr = np.zeros(y_pred_prob.shape[1])
    for prob_idx in range(y_pred_prob.shape[1]):
        score_arr[prob_idx] = average_precision_score(y_true_multilabel_indicator[:, prob_idx], y_pred_prob[:, prob_idx], pos_label=y_true_multilabel_indicator[:, prob_idx][0])
        
    return np.mean(score_arr)
  elif target_type == 'binary':
    return average_precision_score(y_true, y_pred_prob[:, 1], pos_label=y_true.iloc[0])
  else:
    raise Exception('Invalid target type')

def get_fpr_tpr(y_true, y_true_multilabel_indicator, y_pred_prob, target_type):
  if target_type == 'multiclass':
    fpr_arr = np.zeros(y_pred_prob.shape[1])
    tpr_arr = np.zeros(y_pred_prob.shape[1])
    auc_arr = np.zeros(y_pred_prob.shape[1])
    for prob_idx in range(y_pred_prob.shape[1]):
        fpr, tpr, thresholds = roc_curve(y_true_multilabel_indicator[:, prob_idx], y_pred_prob[:, prob_idx], pos_label=y_true_multilabel_indicator[:, prob_idx][0])
        auc_arr[prob_idx] = auc(fpr, tpr)

        for idx, th in enumerate(thresholds):
          if th < 0.5:
            fpr_arr[prob_idx] = fpr[idx - 1]
            tpr_arr[prob_idx] = tpr[idx - 1]
  
    return np.mean(fpr_arr), np.mean(tpr_arr), np.mean(auc_arr)
  elif target_type == 'binary':
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_prob[:, 1], pos_label=y_true.iloc[0])
    
    for idx, th in enumerate(thresholds):
      if th < 0.5:
        return fpr[idx - 1], tpr[idx - 1], auc(fpr, tpr)
    
    raise Exception('Invalid roc curve')
  else:
    raise Exception('Invalid target type')
  
def get_multilabel_indicator(y):
  class_encoder = OneHotEncoder(handle_unknown='ignore')
  y = y.values
  y = y.reshape(-1, 1)
  class_encoder.fit(y)
  y = class_encoder.transform(y).toarray()

  return y

def run_clf(clf_name, dataset_name, X, y, random_state):
  res = None
  print('Testing {0} on {1}...'.format(clf_name, dataset_name))
  kf = StratifiedKFold(n_splits=10)
  t_of_target = type_of_target(y)
  y_multilabel_indicator = get_multilabel_indicator(y)
  roc_auc_sum = 0
  
  for idx, (train_index, test_index) in enumerate(kf.split(X, y)):
      print('Testing Fold {0}'.format(idx + 1))
      X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
      y_train, y_test = y.iloc[train_index], y.iloc[test_index]
      y_train_multilabel_indicator, y_test_multilabel_indicator = y_multilabel_indicator[train_index, :], y_multilabel_indicator[test_index, :]
      clf = get_new_clf(clf_name, t_of_target, random_state)
      inf_curr_time = time.perf_counter()
      clf.fit(X_train, y_train)
      inf_elasped_time = time.perf_counter()
      y_pred = clf.predict(X_test)
      pred_curr_time = time.perf_counter()
      y_pred_1000 = clf.predict(X_test.iloc[:1000, :])
      pred_elasped_time = time.perf_counter()
      y_pred_prob = clf.predict_proba(X_test)
      fpr, tpr, roc_auc = get_fpr_tpr(y_test, y_test_multilabel_indicator, y_pred_prob, t_of_target)
      accuracy = clf.score(X_test, y_test)
      print('Done! Accuracy: {0}'.format(accuracy))
      roc_auc_sum += roc_auc
      df = pd.DataFrame({
        'Dataset Name': [dataset_name],
        'Algorithm Name': [clf_name],
        'Cross Validation [1-10]': [idx + 1],
        'Hyper-Parameters Values' : [clf.best_params_],
        'Accuracy': [accuracy_score(y_test, y_pred)],
        'TPR': [tpr],
        'FPR': [fpr],
        'Precision': [precision_score(y_test, y_pred, average='macro')],
        'AUC': [roc_auc],
        'PR-Curve': [get_average_precision_score(y_test, y_test_multilabel_indicator, y_pred_prob, t_of_target)],
        'Training Time(time.perf_counter())': ['{:.2f}'.format(inf_elasped_time - inf_curr_time)],
        'Inference Time(time.perf_counter()': ['{:.2f}'.format(pred_elasped_time - pred_curr_time)]
      })

      if res is None:
        res = df
      else:
        res = res.append(df, ignore_index=True)
    
  return res, np.mean(roc_auc_sum)

def get_res_ranks_and_res_winner_rows(dataset_name, roc_auc_means_dtype, roc_auc_means):
  arr = np.array(roc_auc_means, dtype=roc_auc_means_dtype)
  arr = np.sort(arr, order='roc_auc_mean')
  arr = np.flip(arr)
  res_ranks_row_dict = {
      'Dataset Name': [dataset_name]
  }
  res_winner_row_dict = {
      'Dataset Name': [],
      'Algorithm Name': [],
      'Winner': []
  }

  for idx, (clf_name, _) in enumerate(arr):
    clf_name = str(clf_name)
    clf_name = clf_name.replace('b\'', '')
    clf_name = clf_name.replace('\'', '')
    res_ranks_row_dict[clf_name] = [idx + 1]
    res_winner_row_dict['Dataset Name'].append(dataset_name)
    res_winner_row_dict['Algorithm Name'].append(clf_name)
    res_winner_row_dict['Winner'].append(idx == 0)
  
  return pd.DataFrame(res_ranks_row_dict), pd.DataFrame(res_winner_row_dict)


def run_clfs_on_all_datasets(clfs_names, datasets_path_name, num_datasets, random_state, force_resume=False):
  res = None
  res_ranks = None
  res_winner = None
  datasets_done = []
  try:
    res = pd.read_csv('results.csv')
    res_ranks = pd.read_csv('results_ranks.csv')
    res_winner = pd.read_csv('results_winner.csv')
    datasets_done = pickle.load(open('datasets_done.p', 'rb'))
    print('Resuming...')
  except:
    print('Unable to resume, not resuming...')
    if force_resume:
      return
    print('Starting fresh...')
  roc_auc_means_dtype = [('clf_name', 'S20'), ('roc_auc_mean', float)]
  count = 0
  
  for dataset_path, X, y in get_all_datasets(datasets_path_name):
    dataset_name = dataset_path.split('/')[-1]
    dataset_name = dataset_name.replace('.csv', '')
    print('Starting on dataset {0}...'.format(dataset_name))
    try:
      if dataset_name in datasets_done:
        print('Skipping {0}, done already...'.format(dataset_name))
        continue
      roc_auc_means = []
      for idx, clf_name in enumerate(clfs_names):
        results, roc_auc_mean = run_clf(clf_name, dataset_name, X, y, random_state)
        roc_auc_means.append((clf_name, roc_auc_mean))

        if res is None:
          res = results
        else:
          res = res.append(results, ignore_index=True)

      res_ranks_row, res_winner_row = get_res_ranks_and_res_winner_rows(dataset_name, roc_auc_means_dtype, roc_auc_means)
      if res_ranks is None:
        res_ranks = res_ranks_row
        res_winner = res_winner_row
      else:
        res_ranks = res_ranks.append(res_ranks_row, ignore_index=True)
        res_winner = res_winner.append(res_winner_row, ignore_index=True)

      count += 1
      if count >= num_datasets:
        break

      generate_resuls_csv(res, 'results.csv')
      generate_resuls_csv(res_ranks, 'results_ranks.csv')
      generate_resuls_csv(res_winner, 'results_winner.csv')
      print('Done on dataset {0}...'.format(dataset_name))

      datasets_done.append(dataset_name)
      pickle.dump(datasets_done, open('datasets_done.p', 'wb'))
      print(datasets_done)
    except:
      print('*** Error on dataset {0}...'.format(dataset_name))
      res = pd.read_csv('results.csv')
      res_ranks = pd.read_csv('results_ranks.csv')
      res_winner = pd.read_csv('results_winner.csv')
  
  return res, res_ranks, res_winner

def generate_resuls_csv(results, target_path):
  results.to_csv(target_path, index=False)


results, results_ranks, results_winner = run_clfs_on_all_datasets(['MadaBoost', 'AdaBoost', 'LogitBoost'], './classification_datasets/', 150, 42, force_resume=True)
generate_resuls_csv(results, 'results.csv')
generate_resuls_csv(results_ranks, 'results_ranks.csv')
generate_resuls_csv(results_winner, 'results_winner.csv')


