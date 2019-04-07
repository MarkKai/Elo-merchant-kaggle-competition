# ----------------------------------------------------------------------------------------
# The is a utilities function .py file
# It will include some useful functions
# that will be used in main ipynb file

# Reference:
#     https://www.kaggle.com/fabiendaniel/elo-world
#     https://www.kaggle.com/fabiendaniel/selecting-features/notebook
#     https://www.kaggle.com/fabiendaniel/hyperparameter-tuning
# ----------------------------------------------------------------------------------------

# define the function to calculate the feature importance
def get_feature_importance(data, target, clf='lightgbm', shuffle=True):
    
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
    
        # model parameters
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


# define function to display imoprtance distribution
def display_importance_dist(importance_test_df, importance_baseline_df, feature):
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # plot window
    plt.figure(figsize=(16,6))
    gs = plt.GridSpec(nrows=1, ncols=2)
    
    # gain importance
    ax = plt.subplot(gs[0, 0])
    _ = sns.distplot(a=importance_test_df[importance_test_df['feature']==feature, 'importance_gain'].values, bins=20, hist=True, kde=True, label='Gain importance distribution', ax=ax)
    ax.vlines(x=importance_baseline_df[importance_baseline_df['feature']==feature, 'importance_gain'].values, ymin=0, ymax= _.get_ylim()[1], linewidth=5, colors='r', label='base line')
    ax.legend()
    plt.title('Gain importance of {:s}'.format(feature.upper()))
    plt.xlabel('Gain importance distribution for {:s}'.format(feature.upper()))
    
    # split importance
    ax = plt.subplot(gs[0, 1])
    _ = sns.distplot(a=importance_test_df[importance_test_df['feature']==feature, 'importance_split'].values, bins=20, hist=True, kde=True, label='Split importance distribution', ax=ax)
    ax.vlines(x=importance_baseline_df[importance_baseline_df['feature']==feature, 'importance_split'].values, ymin=0, ymax= _.get_ylim()[1], linewidth=5, colors='r', label='base line')
    ax.legend()
    plt.title('Split importance of {:s}'.format(feature.upper()))
    plt.xlabel('Split importance distribution for {:s}'.format(feature.upper()))

    

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




