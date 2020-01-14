#%%import pandas as pd 
import lightgbm as lg 
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE, RFECV
import imblearn as imb
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, StackingClassifier, ExtraTreesClassifier,VotingClassifier
import sklearn.preprocessing as prep 
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_validate
from category_encoders import TargetEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
import numpy as np 
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline, FeatureUnion, make_pipeline
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.decomposition import PCA
from imblearn import over_sampling, ensemble, combine, under_sampling
from imblearn import pipeline as imb_pipe
import mlens.ensemble as ens 
from mlens.metrics import make_scorer
from hyperopt import STATUS_OK, hp
from hyperopt.pyll.stochastic import sample
from keras.models import Sequential, Model
from keras.layers.core import Dense, Activation, Reshape, Flatten, Dropout
from keras.layers import Embedding, Input, Concatenate
from keras.utils import to_categorical

def remove_nas(df):
    one_hot_cols = []
    one_hot_num_cols = []
    target_cols = []
    emb_cols = []

    data = df.copy(deep = True)
    numeric_cols = list(data._get_numeric_data().columns)
    #use our numerical columns to find out our categorical columns
    cat_cols = list(set(data.columns) - set(numeric_cols))

    #check column cardinality even if they are numeric
    #cats first
    for c in cat_cols: 
        n_unique = data[c].nunique()
        if(n_unique == 2): #binary columns
            one_hot_cols.append(c) #apply one-hot
        elif(n_unique <= 10000): #target if the number of occurences is lesser than 100
            target_cols.append(c)
        else: 
            emb_cols.append(c) #embedding columns

    for c in numeric_cols: 
        n_unique = data[c].nunique()
        if(n_unique == 2): #binary columns
            one_hot_num_cols.append(c) #apply one-hot
        elif (n_unique < 15): #still a category?
            target_cols.append(c) #map them as target to increase expressivity
    
    #drop from numerics the ones categorized as one-hot or target
    numeric_cols = list(set(numeric_cols) - set(one_hot_num_cols))
    numeric_cols = list(set(numeric_cols) - set(target_cols))

    for c in one_hot_cols:
        if data[c].isna().sum() > 0:
            data[c] = data[c].fillna(data[c].mode()[0])

    for c in target_cols:
        if data[c].isna().sum() > 0:
            data[c] = data[c].fillna(data[c].mode()[0])

    for c in one_hot_num_cols:
        if data[c].isna().sum() > 0:
            data[c] = data[c].fillna(data[c].mode()[0]) #mode because we are looking for encoding with one-hot
    
    return data, one_hot_cols, one_hot_num_cols, numeric_cols, target_cols, emb_cols

def create_pipeline(one_hot_cols, one_hot_num_cols, numeric_cols, target_cols): 
    cat_one_hot_pipe = make_pipeline(
                                    (prep.OneHotEncoder(handle_unknown='ignore', dtype=np.int, sparse=False)) 
                                    #, (prep.RobustScaler())
                                    )

    cat_one_hot_num_pipe = make_pipeline(
                                        (prep.OneHotEncoder(handle_unknown='ignore', dtype=np.int, sparse=False))
                                        #, (prep.RobustScaler())
                                        )

    num_pipe = make_pipeline(
                            (SimpleImputer(strategy="mean"))
                            )

    cat_target_pipe = make_pipeline(
                                    (TargetEncoder(cols = target_cols, drop_invariant=True))
                                    #, (prep.RobustScaler())
                                    )

    prep_pipe = ColumnTransformer(
        [
            ('oh', cat_one_hot_pipe, one_hot_cols), 
            ('oh_num', cat_one_hot_num_pipe, one_hot_num_cols),
            ('cat_target', cat_target_pipe, target_cols)
        ]
        #, n_jobs = -1
    )

    return prep_pipe

def get_roc_curve(X_train, y_train, X_test, y_test, model): 
    probs = model.predict_proba(X_test)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # calculate AUC
    auc = roc_auc_score(y_test, probs)
    print('AUC: %.3f' % auc)
    # calculate roc curve
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    # plot no skill
    plt.plot([0, 1], [0, 1], linestyle='--', color = "red")
    # plot the precision-recall curve for the model
    plt.plot(fpr, tpr, marker='.', color = "blue")
    # show the plot
    plt.show()

def build_and_fit_pipeline(
                            X_train_k, y_train_k, one_hot_cols, one_hot_num_cols, numeric_cols, target_cols, emb_cols, 
                            p_n_estimators = 1000, p_learning_rate = 0.1, p_max_depth = -1, p_reg_alpha = 0.5, 
                            p_reg_lambda = 0.5, p_num_leaves = 100, p_min_child_samples = 5): 

    lr = LogisticRegression(C = 0.2, max_iter = 6000, n_jobs= 4, class_weight='balanced')

    ada = AdaBoostClassifier(n_estimators=300, learning_rate=0.1, random_state=42)

    #base estimators for stacking
    lgb_1 = lg.LGBMClassifier(n_estimators = p_n_estimators, learning_rate=p_learning_rate,
        metric = 'auc', importance_type='gain', max_depth = p_max_depth, reg_alpha=p_reg_alpha, 
        reg_lambda=p_reg_lambda, num_leaves=p_num_leaves,min_child_samples = p_min_child_samples, 
        class_weight = 'balanced', n_jobs = 4)

    lgb_2 = lg.LGBMClassifier(n_estimators = p_n_estimators, learning_rate=p_learning_rate,
        metric = 'auc', importance_type='gain', max_depth = p_max_depth, reg_alpha=p_reg_alpha, 
        reg_lambda=p_reg_lambda, num_leaves=p_num_leaves,min_child_samples = p_min_child_samples, 
        class_weight = 'balanced', n_jobs = 4)

    lgb_3= lg.LGBMClassifier(n_estimators = p_n_estimators, learning_rate=p_learning_rate,
        metric = 'auc', importance_type='gain', max_depth = p_max_depth, reg_alpha=p_reg_alpha, 
        reg_lambda=p_reg_lambda, num_leaves=p_num_leaves,min_child_samples = p_min_child_samples, 
        class_weight = 'balanced', n_jobs = 4)

    #train a stacking ensemble
    classifiers = [
        ('lr', lr)
        ,('lightgbm_1', lgb_1)
        ,('ada', ada)
        , ('lightgbm_2', lgb_2)
        , ('lightgbm_3', lgb_3)
    ]

    #score = make_scorer(roc_auc_score, greater_is_better = True)
    clf = StackingClassifier(
            estimators = classifiers, passthrough=True, final_estimator=lr, 
            n_jobs = 2 #not to overflow our 
            , cv = 3
    )

    #create pipelines
    prep_pipe = create_pipeline(one_hot_cols, one_hot_num_cols, numeric_cols, target_cols)

    #pipeline provided by imbalance learn library, the standard sklearn wouldn't work
    #because it needs an object implementing fit and fit_transform (not the case)
    final_pipeline = Pipeline(
        [
            ('preprocess', prep_pipe), #parallel process of the dataset
            #('PCA', pca_transformer),
            #('imbalance_sampler', sampler),
            #('poly interactions', poly),
            #('RFE', rfe_selector),
            ('classifier', clf)
        ]
    )

    #fit data and model simultanously
    final_pipeline.fit(X = X_train_k, y = y_train_k)

    return final_pipeline


def featurize(train, test, one_hot_cols, one_hot_num_cols, numeric_cols, target_cols):
    feats = []
    added_cols = []
    numeric_cols = train._get_numeric_data().columns
    #use our numerical columns to find out our categorical columns
    cat_cols = one_hot_cols.append(one_hot_num_cols)
    cat_cols = cat_cols.append(target_cols)

    for cat in cat_cols: #not nec
        print('creating features for ' + cat)
        #create the grouping object for each categorical feature we've got
        # FROM TRAINING SET
        group_by_feat = train.groupby(by=cat)
        for num_feat in numeric_cols:
            #go over our numeric features to generate synthetic features
            col_names = [cat, 'mean_' + num_feat + '_by_' + cat, 'std_' + num_feat + '_by_' + cat, 
                            'max_' + num_feat + '_by_' + cat, 'min_' + num_feat + '_by_' + cat]

            #create mean, std, max and min 
            df_grouped = group_by_feat[num_feat].agg(['mean', 'std', 'max', 'min']).reset_index()
            #add feature names
            df_grouped.columns = col_names

            added_cols.append(col_names)
            #for train data
            train = pd.merge(left = train, right=df_grouped, how = 'left', on=cat, 
                                suffixes = ('', '_feat'))

            #number of sigmas the amount is deviated from the mean
            train[num_feat + '_sigmas_on_' + cat] = round((train[num_feat] - train['mean_' + num_feat + '_by_' + cat]) / train['std_' + num_feat + '_by_' + cat], 2)
            train[num_feat + '_amplitude_on_' + cat] = round(train['max_' + num_feat + '_by_' + cat] - train['min_' + num_feat + '_by_' + cat], 2)
            
            print('shape of test: ' + str(train.shape))

            #join data from train set into test set to avoid data leakage
            test = pd.merge(left = test, right=df_grouped, how = 'left', on=cat, 
                                suffixes = ('', '_feat'))
            
            #number of sigmas the amount is deviated from the mean
            test[num_feat + '_sigmas_on_' + cat] = round(abs((test[num_feat] - test['mean_' + num_feat + '_by_' + cat])) / test['std_' + num_feat + '_by_' + cat], 2)
            test[num_feat + '_amplitude_on_' + cat] = round(test['max_' + num_feat + '_by_' + cat] - test['min_' + num_feat + '_by_' + cat], 2)
            
            print('shape of test: ' + str(test.shape))
    
    return train, test

