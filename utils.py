# ----------------------------------------------------------------------------------------
# The is a utilities function .py file
# It will include some useful functions
# that will be used in main ipynb file

# Reference:
#     https://www.kaggle.com/fabiendaniel/elo-world
#     https://www.kaggle.com/ogrellier/feature-selection-with-null-importances
#     https://www.kaggle.com/fabiendaniel/hyperparameter-tuning
# ----------------------------------------------------------------------------------------

# define the function to calculate the feature importance
def get_feature_importance(data, target, clf='lightgbm', shuffle=False):
    '''
    Parameters
    ----------
    data: input dataset, type of dataframe
    
    target: input target dataset, type of series
    
    clf: the name of model want to use, type of string
    
    shuffle: whether to shuffle target dataset (for getting null importance)
    
    Return
    ------
    importance_df: importance of each features, type of dataframe, shape(n_feature, n_importance)
    '''
    
    # feature list
    train_features = [feature for feature in data.columns.values if feature not in ['target', 'card_id', 'first_active_month']]
    categorical_features = [feature for feature in train_features if 'feature_' in feature]
    
    # shuffle the data
    y = target.copy().sample(frac=1.0) if shuffle else target.copy()
    
    # using lightgbm
    if clf == 'lightgbm':
        import lightgbm as lgb
        import pandas as pd
    
        # construct training date
        train_data = lgb.Dataset(data=data[train_features],\
                                 label=y,\
                                 free_raw_data=False)
    
        # model hyperparameters
        lgb_params = {
            'num_leaves': 129,
            'min_data_in_leaf': 148,
            'objective': 'regression',
            'max_depth': 9,
            'learning_rate': 0.005,
            'min_child_samples': 24,
            'boosting': 'gbdt',
            'feature_fraction': 0.7202,
            'bagging_freq': 1,
            'bagging_fraction': 0.8125,
            'bagging_seed': 11,
            'metric': 'rmse',
            'lambda_l1': 0.3468,
            'random_state': 133,
            'verbosity': -1
        }
    
        # training the model
        clf_lgb = lgb.train(params=lgb_params,\
                            train_set=train_data,\
                            num_boost_round=850)
        
        # calculate importance
        importance_df = pd.DataFrame()
        importance_df['feature'] = list(train_features)
        importance_df['importance_gain'] =\
        clf_lgb.feature_importance(importance_type='gain')
        importance_df['importance_split'] =\
        clf_lgb.feature_importance(importance_type='split')

        return importance_df
    
    if clf == 'catboost':
        from catboost import train, Pool, EFstrType
        import pandas as pd
        
        # construct training data
        train_data = Pool(data=data[train_features],\
                          label=y)
        
        # model hyperparameters
        cat_params = {
            'loss_function': 'RMSE',
            'learning_rate': 0.02,
            'early_stopping_rounds': 400,
            'border_count': 254,
            'task_type': 'GPU',
            'one_hot_max_size': 6,
            'depth': 11,
            'l2_leaf_reg': 1.0,
            'random_strength': 1.9574,
            'bagging_temperature': 20.9049
        }
        
        # training the model
        clf_cat = train(pool=train_data,\
                        params=cat_params,\
                        verbose=False,\
                        iterations=1000)
        
        # calculate feature importance
        importance_df = pd.DataFrame()
        importance_df['feature'] = list(train_features)
        importance_df['PredictionValuesChange'] =\
        clf_cat.get_feature_importance(type='PredictionValuesChange')
        
        return importance_df


# define function to display imoprtance distribution
def display_importance_dist(importance_null_df, importance_baseline_df, feature, clf='lightbgm'):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    if clf == 'lightbgm':
        # plot window
        plt.figure(figsize=(16,6))
        gs = plt.GridSpec(nrows=1, ncols=2)

        # gain importance
        ax = plt.subplot(gs[0, 0])
        _ = sns.distplot(a=importance_null_df.loc[importance_null_df['feature']==feature, 'importance_gain'].values, bins=20, hist=True, kde=True, label='Gain importance distribution', ax=ax)
        ax.vlines(x=importance_baseline_df.loc[importance_baseline_df['feature']==feature, 'importance_gain'].values, ymin=0, ymax= _.get_ylim()[1], linewidth=5, colors='r', label='base line')
        ax.legend()
        plt.title('Gain importance of {:s}'.format(feature.upper()))
        plt.xlabel('Gain importance distribution for {:s}'.format(feature.upper()))

        # split importance
        ax = plt.subplot(gs[0, 1])
        _ = sns.distplot(a=importance_null_df.loc[importance_null_df['feature']==feature, 'importance_split'].values, bins=20, hist=True, kde=True, label='Split importance distribution', ax=ax)
        ax.vlines(x=importance_baseline_df.loc[importance_baseline_df['feature']==feature, 'importance_split'].values, ymin=0, ymax= _.get_ylim()[1], linewidth=5, colors='r', label='base line')
        ax.legend()
        plt.title('Split importance of {:s}'.format(feature.upper()))
        plt.xlabel('Split importance distribution for {:s}'.format(feature.upper()))
    
    if clf == 'catboost':
        # plot window
        plt.figure(figsize=(8,6))
        _ = sns.distplot(a=importance_null_df.loc[importance_null_df['feature']==feature, 'PredictionValuesChange'].values, bins=20, hist=True, kde=True, label='PredictionValuesChange imoprtance distribution')
        plt.vlines(x=importance_baseline_df.loc[importance_baseline_df['feature']==feature, 'PredictionValuesChange'].values, ymin=0, ymax= _.get_ylim()[1], linewidth=5, colors='r', label='base line')
        plt.legend()
        plt.title('PredictionValuesChange importance of {:s}'.format(feature.upper()))
        plt.xlabel('PredictionValuesChange importance distribution for {:s}'.format(feature.upper()))
    
    return
    

# reduce memery usage
def reduce_mem_usage(df):
    ''' source: https://www.kaggle.com/fabiendaniel/hyperparameter-tuning '''
    import numpy as np
    
    
    # initial memery usage
    start_mem = df.memory_usage().sum() / 1024**2
    
    # recude memeery usage for each feature
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            
            # dtype is int
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df.loc[:, col] = df.loc[:, col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df.loc[:, col] = df.loc[:, col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df.loc[:, col] = df.loc[:, col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df.loc[:, col] = df.loc[:, col].astype(np.int64)  
            # dtype is float
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df.loc[:, col] = df.loc[:, col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df.loc[:, col] = df.loc[:, col].astype(np.float32)
                else:
                    df.loc[:, col] = df.loc[:, col].astype(np.float64)
    # after reduction memery usage
    end_mem = df.memory_usage().sum() / 1024**2
    print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def LightGBM_tuning(X_train, y_train, kfold=6):
    '''
    LightGBM model hyperparameters tuning, use baye_opt to cross-validate entire training dataset.
    @ tuning hyperparameters:
        feature_fraction
        bagging_fraction
        lambda_l1
        max_depth
        min_data_in_leaf
        num_leaves
    @ default hyperparameters:
        bagging_freq = 1
        bagging_seed = 11
        boosting = 'gbdt'
        learning_rate: 0.005
        
    Parameters
    ----------
    X_train: feature dataframe
    
    y_train: target series
    
    Return
    ------
    dict: diction of tuning hyperparameters of lightGBM
    '''
    
    import lightgbm as lgb
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    import numpy as np
    import gc
    from bayes_opt.observer import JSONLogger
    from bayes_opt.event import Events
    from bayes_opt import BayesianOptimization
    
    X_train = X_train
    y_train = y_train
    features = [feature for feature in X_train.columns \
                if feature not in ['card_id', 'first_active_month']]
    categorical_features = [feature for feature in features \
                            if 'feature_' in feature]
    folds = KFold(n_splits=kfold, shuffle=True, random_state=133)
    y_val = np.zeros(y_train.shape)
    bayes_opt_params = {
        'feature_fraction': (0.1, 1.0),
        'bagging_fraction': (0.1, 1.0),
        'lambda_l1': (0, 6),
        'max_depth': (4, 20),
        'min_data_in_leaf': (10, 300),
        'num_leaves': (5, 300),
    }
    
    # define the croos-validation functions which returns object score(-rmse)
    # then use bayesian optimizers to tuning the object score
    def cv_helper(max_depth,\
                  num_leaves,\
                  min_data_in_leaf,\
                  feature_fraction,\
                  bagging_fraction,\
                  lambda_l1):
        
        for train_idxs, val_idxs in folds.split(X_train.values, y_train.values):
            
            # training set
            train_data = lgb.Dataset(data=X_train.iloc[train_idxs][features],\
                                     label=y_train.iloc[train_idxs],\
                                     categorical_feature=categorical_features)

            # validation set
            val_data = lgb.Dataset(data=X_train.iloc[val_idxs][features],\
                                   label=y_train.iloc[val_idxs],\
                                   categorical_feature=categorical_features)
            # hyperparameters
            params = {
                'objective': 'regression',
                'metric': 'rmse',
                'lambda_l1': lambda_l1,
                'num_leaves': int(num_leaves),
                'min_data_in_leaf': int(min_data_in_leaf),
                'max_depth': int(max_depth),
                'feature_fraction': feature_fraction,
                'bagging_fraction': bagging_fraction,
                'bagging_freq': 1,
                'bagging_seed': 11,
                'boosting': 'gbdt',
                'learning_rate': 0.005,
                'verbosity': 1
            }
            
            # classifier
            clf = lgb.train(params=params,\
                            train_set=train_data,\
                            num_boost_round=10000,\
                            valid_sets=[train_data, val_data],\
                            verbose_eval=200,\
                            early_stopping_rounds=200)
            
            # prediction of validation
            y_val[val_idxs] = clf.predict(X_train.iloc[val_idxs][features],\
                                          num_iteration=clf.best_iteration)
            
        return -mean_squared_error(y_true=y_train, y_pred=y_val)**0.5
    
    logger = JSONLogger(path="bayes_opt_log/lightGBM_logs.json")
    LGB_bayes_opt = BayesianOptimization(cv_helper, pbounds=bayes_opt_params)
    LGB_bayes_opt.subscribe(Events.OPTMIZATION_STEP, logger)
    LGB_bayes_opt.maximize(init_points=4,\
                       n_iter=20,\
                       acq='ei',\
                       xi=0.0)
    
    return LGB_bayes_opt.max['params']

def Catboost_tuning(X_train, y_train, kfold=6):
    '''
    Catboost model hyperparameters tuning, use baye_opt to cross-validate entire training dataset.
    @ tuning hyperparameters:
        one_hot_max_size: if required int
        depth: 6 ~ 10 int
        l2_leaf_reg: positive value 1 ~ 30
        random_strength: 1 ~ 30
        bagging_temperature: 0 ~ 1000
        
    @ default hyperparameters:
        NUMER_OF_TREES:
            iterations: 10000
            use_best_model = True
            eval_metric = 'RMSE'
            eval_set = Pool()
        learning_rate = 0.02
        border_count = 254
        
    Parameters
    ----------
    X_train: feature dataframe
    
    y_train: target series
    
    Return
    ------
    dict: diction of tuning hyperparameters of Catboost
    '''
    
    from catboost import train, Pool
    from sklearn.model_selection import KFold
    import numpy as np
    import gc
    from bayes_opt.observer import JSONLogger
    from bayes_opt.event import Events
    from bayes_opt import BayesianOptimization
    
    X_train = X_train
    y_train = y_train
    features = [feature for feature in X_train.columns \
                if feature not in ['card_id', 'first_active_month']]
    categorical_features = [feature for feature in features \
                            if 'feature_' in feature]
    folds = KFold(n_splits=kfold, shuffle=True, random_state=133)
    catboost_opt_params = {
        'one_hot_max_size': (0, 6),
        'depth': (5, 11),
        'l2_leaf_reg': (1, 30),
        'random_strength': (1, 30),
        'bagging_temperature': (0, 1000)
    }
    
    def cv_helper(one_hot_max_size,\
                  depth,\
                  l2_leaf_reg,\
                  random_strength,\
                  bagging_temperature):
        
        # entire date for evaluate clf training performance
        all_data = Pool(data=X_train[features],\
                        label=y_train,\
                        cat_features=categorical_features)
        # validation RMSE
        RMSE = []
        
        for train_idxs, val_idxs in folds.split(X_train.values, y_train.values):
            
            # training set
            train_data = Pool(data=X_train.iloc[train_idxs][features],\
                              label=y_train.iloc[train_idxs],\
                              cat_features=categorical_features)

            # validation set
            val_data = Pool(data=X_train.iloc[val_idxs][features],\
                            label=y_train.iloc[val_idxs],\
                            cat_features=categorical_features)
            # hyperparameters
            params = {
                'eval_metric': 'RMSE',
                'use_best_model': True,
                'loss_function': 'RMSE',
                'learning_rate': 0.02,
                'early_stopping_rounds': 400,
                'border_count': 254,
                'task_type': 'GPU',
                'one_hot_max_size': int(one_hot_max_size),
                'depth': int(depth),
                'l2_leaf_reg': l2_leaf_reg,
                'random_strength': random_strength,
                'bagging_temperature': bagging_temperature
            }
            
            # classifier
            clf = train(pool=train_data,\
                        params=params,\
                        verbose=200,\
                        iterations=10000,\
                        eval_set=all_data)
            
            # add current fold RMSE on all_data
            RMSE.append(clf.best_score_['validation_0']['RMSE'])
            
        return -np.mean(np.array(RMSE))
    
    logger = JSONLogger(path="bayes_opt_log/catBoost_logs.json")
    CAT_bayes_opt = BayesianOptimization(cv_helper, pbounds=catboost_opt_params)
    CAT_bayes_opt.subscribe(Events.OPTMIZATION_STEP, logger)
    CAT_bayes_opt.maximize(init_points=4,\
                       n_iter=20,\
                       acq='ei',\
                       xi=0.0)
    
    return CAT_bayes_opt.max['params']

def save_params(params_dict, path):
    ''' save model hyperparameters to pickle file '''
    import pickle
    f = open(path, 'wb')
    pickle.dump(params_dict, f)
    f.close()
    
    return

def load_params(path):
    ''' load model hyperparameters from pickle file '''
    import pickle
    f = open(path, 'rb')
    _dict = pickle.load(f)
    f.close()
    return _dict

def get_feature_PIMP(importance_null_df, importance_pvalue_df):
    '''
    This method is an alternative implementation of paper: Permutation importance: a corrected feature importance measure, Altmann et al. (2010)
    '''
    import pandas as pd
    import numpy as np
    
    _feature_PIMP = []
    
    # traversal each feature
    for feature in importance_null_df['feature'].unique():
        
        # traversal each type of importance
        _PIMP_list = [feature]
        for type_of_imp in importance_null_df.columns[1:]:
            
            importance_null = \
            importance_null_df[importance_null_df['feature']==\
                              feature][type_of_imp].values
            importance_pvalue =\
            importance_pvalue_df[importance_pvalue_df['feature']==\
                             feature][type_of_imp].values
            _PIMP =\
            np.log(1e-10 + importance_pvalue/(0.1+np.percentile(importance_null, 75)))
            _PIMP_list.extend([_PIMP[0]])
        
        _feature_PIMP.append(_PIMP_list)
            
    _feature_PIMP = pd.DataFrame(data=_feature_PIMP,\
                                columns=importance_null_df.columns)
    
    return _feature_PIMP

def visualize_feature_score(feature_score, num_feature):
    '''
    Visualize feature score in ascending order, with specific number of features
    
    Parameter
    ---------
    feature_score: the score of each feature, type of dataframe
    
    num_feature: the number of feature want to visualize, type of int
    '''
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    _num_imp = len(feature_score.columns) - 1
    plt.figure(figsize=(10, int(num_feature*0.8)))
    gs = plt.GridSpec(_num_imp, 1)
    
    for idx in range(_num_imp):
        ax = plt.subplot(gs[idx, 0])
        sns.barplot(x=feature_score.columns[idx+1],\
                    y=feature_score.columns[0],\
                    data=feature_score.sort_values(by=feature_score.columns[idx+1],\
                                                   axis=0,\
                                                   ascending=False)\
                    .iloc[:num_feature],\
                    ax=ax)
        _ = plt.title('PIMP of {} of features'.format(feature_score.columns[idx+1]),\
                      fontdict={'fontweight':'bold', 'fontsize':14})
    
    return

def lgb_cv_train(X_train, y_train, params, kfold):
    '''
    taking feature and target dataset and using cross-validation method to train model, return list of classifiers, length of list is the number of clfs for each fold.
    '''
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split, KFold
    
    features = [feature for feature in X_train.columns if feature not in ['target', 'card_id', 'first_active_month']]
    categorical_features = [feature for feature in features if 'feature_' in feature]
    folds = KFold(n_splits=kfold, shuffle=True, random_state=42)
    clf_list = []
    
    for train_idxs, val_idxs in folds.split(X_train.values, y_train.values):
        
        # training set
        train_set = lgb.Dataset(data=X_train.iloc[train_idxs][features],\
                                label=y_train.iloc[train_idxs],\
                                categorical_feature=categorical_features)
        # validation set
        valid_set = lgb.Dataset(data=X_train.iloc[val_idxs][features],\
                                label=y_train.iloc[val_idxs],\
                                categorical_feature=categorical_features)

        # train clf
        clf = lgb.train(params=params,\
                        train_set=train_set,\
                        num_boost_round=10000,\
                        valid_sets=[train_set, valid_set],\
                        early_stopping_rounds=200,\
                        verbose_eval=100)
        
        # add current clf to clf_list
        clf_list.append(clf)
        
    return clf_list


def cat_cv_train(X_train, y_train, params, kfold):
    '''
    taking feature and target dataset and using cross-validation method to train model, return list of classifiers, length of list is the number of clfs for each fold.
    '''
    from catboost import train, Pool
    from sklearn.model_selection import train_test_split, KFold
    
    features = [feature for feature in X_train.columns if feature not in ['target', 'card_id', 'first_active_month']]
    categorical_features = [feature for feature in features if 'feature_' in feature]
    folds = KFold(n_splits=kfold, shuffle=True, random_state=42)
    clf_list = []
    
    for train_idxs, val_idxs in folds.split(X_train.values, y_train.values):
        
        # training set
        train_set = Pool(data=X_train.iloc[train_idxs][features],\
                         label=y_train.iloc[train_idxs],\
                         cat_features=categorical_features)
        # validation set
        valid_set = Pool(data=X_train.iloc[val_idxs][features],\
                         label=y_train.iloc[val_idxs],\
                         cat_features=categorical_features)

        # train clf
        clf = train(pool=train_set,\
                    params=params,\
                    verbose=100,\
                    iterations=10000,\
                    eval_set=valid_set)
        
        # add current clf to clf_list
        clf_list.append(clf)
        
    return clf_list