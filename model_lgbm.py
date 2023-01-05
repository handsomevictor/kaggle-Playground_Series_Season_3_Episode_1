import xgboost as xgb

params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'eta': 0.1,
    'max_depth': 6,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'min_child_weight': 1,
    'alpha': 0,
    'lambda': 1,
    'seed': 42,
    'nthread': 4,
    'silent': True
}


if __name__ == '__main__':
    pass
