# -*- coding: utf-8 -*-
"""
This script shows how to apply 80-20 holdout train and validate regression model to predict
MOS from the features
"""
import pandas
import scipy.io
import numpy as np
import argparse
import time
import math
import os, sys
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn import preprocessing
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.svm import LinearSVR
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
import scipy.stats
from concurrent import futures
import functools
import warnings
warnings.filterwarnings("ignore")
# ----------------------- Set System logger ------------- #
class Logger:
  def __init__(self, log_file):
    self.terminal = sys.stdout
    self.log = open(log_file, "a")

  def write(self, message):
    self.terminal.write(message)
    self.log.write(message)  

  def flush(self):
    #this flush method is needed for python 3 compatibility.
    #this handles the flush command by doing nothing.
    #you might want to specify some extra behavior here.
    pass


def arg_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_name', type=str, default='RAPIQUE',
                      help='Evaluated BVQA model name.')
  parser.add_argument('--dataset_name', type=str, default='KONVID_1K',
                      help='Evaluation dataset.') 
  parser.add_argument('--feature_file', type=str,
                      default='feat_files/KONVID_1K_RAPIQUE_feats.mat',
                      help='Pre-computed feature matrix.')
  parser.add_argument('--mos_file', type=str,
                      default='mos_files/KONVID_1K_metadata.csv',
                      help='Dataset MOS scores.')
  parser.add_argument('--num_cont', type=int,
                      default=10,
                      help='Number of contents.')
  parser.add_argument('--num_dists', type=int,
                      default=15,
                      help='Number of distortions per content.')
  parser.add_argument('--out_file', type=str,
                      default='result/KONVID_1K_RAPIQUE_SVR_corr.mat',
                      help='Output correlation results')
  parser.add_argument('--log_file', type=str,
                      default='logs/KONVID_1K_RAPIQUE_SVR.log',
                      help='Log files.')
  parser.add_argument('--color_only', action='store_true',
                      help='Evaluate color values only. (Only for YouTube UGC)')
  parser.add_argument('--log_short', action='store_true',
                      help='Whether log short')
  parser.add_argument('--use_parallel', action='store_true',
                      help='Use parallel for iterations.')
  parser.add_argument('--num_iterations', type=int, default=50,
                      help='Number of iterations of train-test splits')
  parser.add_argument('--max_thread_count', type=int, default=4,
                      help='Number of threads.')
  args = parser.parse_args()
  return args

def logistic_func(X, bayta1, bayta2, bayta3, bayta4):
  # 4-parameter logistic function
  logisticPart = 1 + np.exp(np.negative(np.divide(X - bayta3, np.abs(bayta4))))
  yhat = bayta2 + np.divide(bayta1 - bayta2, logisticPart)
  return yhat

def compute_metrics(y_pred, y):
  '''
  compute metrics btw predictions & labels
  '''
  # compute SRCC & KRCC
  SRCC = scipy.stats.spearmanr(y, y_pred)[0]
  try:
    KRCC = scipy.stats.kendalltau(y, y_pred)[0]
  except:
    KRCC = scipy.stats.kendalltau(y, y_pred, method='asymptotic')[0]

  # logistic regression btw y_pred & y
  beta_init = [np.max(y), np.min(y), np.mean(y_pred), 0.5]
  popt, _ = curve_fit(logistic_func, y_pred, y, p0=beta_init, maxfev=int(1e8))
  y_pred_logistic = logistic_func(y_pred, *popt)
  
  # compute  PLCC RMSE
  PLCC = scipy.stats.pearsonr(y, y_pred_logistic)[0]
  RMSE = np.sqrt(mean_squared_error(y, y_pred_logistic))
  return [SRCC, KRCC, PLCC, RMSE]

def formatted_print(snapshot, params, duration):
  print('======================================================')
  print('params: ', params)
  print('SRCC_train: ', snapshot[0])
  print('KRCC_train: ', snapshot[1])
  print('PLCC_train: ', snapshot[2])
  print('RMSE_train: ', snapshot[3])
  print('======================================================')
  print('SRCC_test: ', snapshot[4])
  print('KRCC_test: ', snapshot[5])
  print('PLCC_test: ', snapshot[6])
  print('RMSE_test: ', snapshot[7])
  print('======================================================')
  print(' -- ' + str(duration) + ' seconds elapsed...\n\n')

def final_avg(snapshot):
  def formatted(args, pos):
    mean = np.nanmean(list(map(lambda x: x[pos], snapshot)))
    stdev = np.nanstd(list(map(lambda x: x[pos], snapshot)))
    print('{}: {} (std: {})'.format(args, mean, stdev))

  print('======================================================')
  print('Average training results among all repeated 80-20 holdouts:')
  formatted("SRCC Train", 0)
  formatted("KRCC Train", 1)
  formatted("PLCC Train", 2)
  formatted("RMSE Train", 3)
  print('======================================================')
  print('Average testing results among all repeated 80-20 holdouts:')
  formatted("SRCC Test", 4)
  formatted("KRCC Test", 5)
  formatted("PLCC Test", 6)
  formatted("RMSE Test", 7)
  print('\n\n')

def idx_expand(idx, num_dists):
  idx_out = []
  for ii in idx:
    idx_out.extend(range(ii*num_dists,(ii+1)*num_dists))
  return idx_out

def evaluate_bvqa_one_split(i, X, y, num_cont, num_dists, log_short):
  if log_short:
    print('{} th repeated holdout test'.format(i))
    t_start = time.time()
  # train test split
  idx_train, idx_test = train_test_split(range(num_cont), test_size=0.1875, #0.1875--81.25%,  0.2----75%
      random_state=math.ceil(8.8*i))
  #print('idx_train" ', idx_train)
  #print('idx_test" ', idx_test)
  idx_param_train, idx_param_test = train_test_split(range(len(idx_train)), test_size=0.1875,
       random_state=math.ceil(6.6*i))
  X_train = X[idx_expand(idx_train, num_dists), :]
  X_test = X[idx_expand(idx_test, num_dists), :]
  y_train = y[idx_expand(idx_train, num_dists)]
  y_test = y[idx_expand(idx_test, num_dists)]
  k_folds = 4
  # split_index_cv = [idx // (len(idx_train)//k_folds*num_dists)
  #                   for idx in range(len(idx_train)*num_dists)]  
  split_index_cv = [idx*k_folds // (len(y_train))
                    for idx in range(len(y_train))] 
  pdsplit = PredefinedSplit(test_fold=split_index_cv)
  
  
  
  # grid search CV on the training set
  if X_train.shape[1] <= 2000:
    #print(f'{X_train.shape[1]}-dim features, using SVR')
     #grid search CV on the training set
    param_grid = {'C': np.logspace(1, 10, 10, base=2),
                  'gamma': np.logspace(-8, 1, 10, base=2)}
    grid_s = RandomizedSearchCV(SVR(kernel = 'rbf'), param_grid, cv=pdsplit, n_jobs=-1)
    grid_t = RandomizedSearchCV(SVR(kernel = 'rbf'), param_grid, cv=pdsplit, n_jobs=-1)
    grid_g = RandomizedSearchCV(SVR(kernel = 'rbf'), param_grid, cv=pdsplit, n_jobs=-1)

  else:
    print(f'{X_train.shape[1]}-dim features, using LinearSVR')
    # grid search on liblinear 
    param_grid = {'C': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.],
                  'epsilon': [0.001, 0.01, 0.1, 1., 2.5, 5., 10.]}
    grid = RandomizedSearchCV(LinearSVR(), param_grid, cv=pdsplit)

  

  
  X_train_s = X_train[:,0:272]
  X_train_t = X_train[:,272:748] 
  X_train_g = X_train[:,748:1260] 
  #print(X_train_s.shape)
  #print(X_train_t.shape)
  X_test_s = X_test[:,0:272]
  X_test_t = X_test[:,272:748] 
  X_test_g = X_test[:,748:1260]
  

  ## Scaler first
  scaler_s = preprocessing.MinMaxScaler().fit(X_train_s)  
  X_train_s = scaler_s.transform(X_train_s)
  X_test_s = scaler_s.transform(X_test_s)
  
  scaler_t = preprocessing.MinMaxScaler().fit(X_train_t)  
  X_train_t = scaler_t.transform(X_train_t)
  X_test_t = scaler_t.transform(X_test_t)

  scaler_g = preprocessing.MinMaxScaler().fit(X_train_g)  
  X_train_g = scaler_g.transform(X_train_g)
  X_test_g = scaler_g.transform(X_test_g)


 
  # grid search
  grid_s.fit(X_train_s, y_train)
  best_params_s = grid_s.best_params_
  
  grid_t.fit(X_train_t, y_train)
  best_params_t = grid_t.best_params_

  grid_g.fit(X_train_g, y_train)
  best_params_g = grid_g.best_params_
  # init model
  if X_train.shape[1] <= 2000:
    regressor_s = SVR(kernel = 'rbf', C=best_params_s['C'], gamma=best_params_s['gamma'])
    regressor_t = SVR(kernel = 'rbf', C=best_params_t['C'], gamma=best_params_t['gamma'])
    regressor_g = SVR(kernel = 'rbf', C=best_params_g['C'], gamma=best_params_g['gamma'])
    #regressor = RandomForestRegressor(**best_params)
  else:
    regressor = LinearSVR(C=best_params['C'], epsilon=best_params['epsilon'])
  # re-train the model using the best alpha
  regressor_s.fit(X_train_s, y_train)
  regressor_t.fit(X_train_t, y_train)
  regressor_g.fit(X_train_g, y_train)
  # predictions
  y_train_pred_s = regressor_s.predict(X_train_s)
  y_train_pred_t = regressor_t.predict(X_train_t)
  y_train_pred_g = regressor_g.predict(X_train_g)
  y_test_s = regressor_s.predict(X_test_s)
  y_test_t = regressor_t.predict(X_test_t)
  y_test_g = regressor_g.predict(X_test_g)
  # compute metrics
  y_train_pred = [(x+y+z)/3  for x, y, z in zip(y_train_pred_s,y_train_pred_t,y_train_pred_g)]
  y_test_pred = [(x+y+z)/3   for x, y, z in zip(y_test_s,y_test_t,y_test_g)]
  metrics_train = compute_metrics(y_train_pred, y_train)
  metrics_test = compute_metrics(y_test_pred, y_test)
  # print values
  #if not log_short:
  #  t_end = time.time()
  #  formatted_print(metrics_train + metrics_test, best_params, (t_end - t_start))
  return metrics_train, metrics_test
  
def main(args):
  df = pandas.read_csv(args.mos_file, skiprows=[], header=None)
  array = df.values
  if args.dataset_name == 'LIVE_VQC':
      y = array[1:,1]
  elif args.dataset_name == 'KONVID_1K': # for LIVE-VQC & KONVID_1k
      y = array[1:,1]
  elif args.dataset_name == 'YOUTUBE_UGC':
      y = array[1:,4]
  elif args.dataset_name == 'LIVE_VQA':
      y = array[1:,1]
  elif args.dataset_name == 'LIVE_HFR':
      y = array[1:,1]
  elif args.dataset_name == 'ETRI_LIVE':
      y = array[1:,1]
  y = np.array(list(y), dtype=np.float)
  X_mat = scipy.io.loadmat(args.feature_file)
  X = np.asarray(X_mat['STFastFeat'], dtype=np.float)


  '''57 grayscale videos in YOUTUBE_UGC dataset, we do not consider them for fair comparison'''
  if args.color_only and args.dataset_name == 'YOUTUBE_UGC':
      gray_indices = [3,6,10,22,23,46,51,52,68,74,77,99,103,122,136,141,158,173,368,426,467,477,506,563,594,\
      639,654,657,666,670,671,681,690,697,702,703,710,726,736,764,768,777,786,796,977,990,1012,\
      1015,1023,1091,1118,1205,1282,1312,1336,1344,1380]
      gray_indices = [idx - 1 for idx in gray_indices]
      X = np.delete(X, gray_indices, axis=0)
      y = np.delete(y, gray_indices, axis=0)
  ## preprocessing
  X[np.isinf(X)] = np.nan
  imp = SimpleImputer(missing_values=np.nan, strategy='mean').fit(X)
  X = imp.transform(X)
 

  all_iterations = []
  t_overall_start = time.time()
  # 100 times random train-test splits
  if args.use_parallel is True:
    evaluate_bvqa_one_split_partial = functools.partial(
       evaluate_bvqa_one_split, X=X, y=y, num_cont=args.num_cont,
       num_dists=args.num_dists, log_short=args.log_short)
    with futures.ThreadPoolExecutor(max_workers=args.max_thread_count) as executor:
      iters_future = [
          executor.submit(evaluate_bvqa_one_split_partial, i)
          for i in range(1, args.num_iterations)]
      for future in futures.as_completed(iters_future):
        metrics_train, metrics_test = future.result()
        all_iterations.append(metrics_train + metrics_test)
  else:
    for i in range(1, args.num_iterations):
        metrics_train, metrics_test = evaluate_bvqa_one_split(
          i, X, y, args.num_cont, args.num_dists, args.log_short)
        all_iterations.append(metrics_train + metrics_test)

  

  # formatted print overall iterations
  #srocc = [it[4] for it in all_iterations] 
  #sv = pandas.DataFrame(columns=['FAVER-Haar'],data=srocc)
  #sv.to_csv('./boxhaar.csv')
  final_avg(all_iterations)
  print('Overall {} secs lapsed..'.format(time.time() - t_overall_start))
  # save figures
  dir_path = os.path.dirname(args.out_file)
  if not os.path.exists(dir_path):
    os.makedirs(dir_path)
  scipy.io.savemat(args.out_file, 
      mdict={'all_iteration': np.asarray(all_iterations,dtype=np.float)})

if __name__ == '__main__':
  args = arg_parser()
  log_file = args.log_file
  log_dir = os.path.dirname(log_file)
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)
  sys.stdout = Logger(log_file)
  print(args)
  main(args)

'''

python evaluate_bvqa_features_regression.py \
  --model_name BRISQUE \
  --dataset_name LIVE_VQC \
  --feature_file mos_feat_files/KONIQ_10K_BRISQUE_feats.mat \
  --mos_file mos_feat_files/KONIQ_10K_metadata.csv \
  --out_file result/KONIQ_10K_BRISQUE_SVR_corr.mat \
  --use_parallel


'''
