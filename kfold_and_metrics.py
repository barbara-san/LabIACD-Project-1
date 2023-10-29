import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


###########################################################################

def get_k_folds(df: pd.DataFrame, k=10):
    def valid_proportions(folds: list[pd.DataFrame]):
        fix_prop = df['malignancy'].value_counts(normalize=True, sort=False).to_dict()
        for fold in folds:
            fold_prop = fold['malignancy'].value_counts(normalize=True, sort=False).to_dict()
            error = 0
            for k in fold_prop.keys():
                error += (fix_prop[k] - fold_prop[k])**2
            if error > 0.1:
                return False
        return True            

    folds = []
    pid_list = df['patient_id'].unique()
    while True:
        shuffled_pid_list = pd.Series(pid_list).sample(frac=1).to_list()
        for pid in range(0, len(pid_list), len(pid_list)//k):
            pid_fold = shuffled_pid_list[pid : pid+len(pid_list)//k]
            fold_df = df[df['patient_id'].isin(pid_fold)]
            folds.append(fold_df)
        
        if valid_proportions(folds): break
        else: folds = []
    if len(folds) == k+1:
        last_fold = folds[-1]
        for index, pid in enumerate(last_fold['patient_id'].unique()):
            folds[index].append(last_fold[last_fold['patient_id']==pid])
        folds = folds[0:-1]
    
    return folds

###########################################################################

###########################################################################

# returns a dicitonary with pairs (metric_name, list_of_results)
# model must have .fit() and .predict() methods

def k_fold_cv(model, df:pd.DataFrame, k=10, metric_funcs:list=[f1_score, accuracy_score, roc_auc_score]):
    folds = get_k_folds(df, k)
    
    metrics_results = dict((metric_fn.__name__, []) for metric_fn in metric_funcs)

    for test_fold_index in range(len(folds)):
        testing_df = folds[test_fold_index]
        training_df = pd.DataFrame(columns=df.columns)
        for fold_index, fold_df in enumerate(folds):
            if fold_index == test_fold_index:
                continue
            training_df.append(fold_df)

        X_train, y_train = training_df.drop(columns=['malignancy']), training_df['malignancy']
        X_test, y_test = testing_df.drop(columns=['malignancy']), testing_df['malignancy']

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        for metric_fn in metric_funcs:
            metrics_results[metric_fn.__name__].append(metric_fn(y_test, y_pred))
        
    return metrics_results

###########################################################################

###########################################################################

# returns a dicitonary with pairs (metric_name, list_of_results)
# model must be a TF Keras model and must have been compiled

def k_fold_cv_keras(compiled_model, df:pd.DataFrame, k=10, metric_funcs:list=[f1_score, accuracy_score, roc_auc_score], num_epochs=10):
    folds: list[pd.DataFrame] = get_k_folds(df, k)

    metrics_results = dict((metric_fn.__name__, []) for metric_fn in metric_funcs)

    for index, fold in enumerate(folds):
        test_df = fold
        train_df = pd.DataFrame(columns=fold.columns)
        for i in range(len(folds)):
            if i == index: continue
            train_df.append(folds[i])
        
        x_train, y_train = train_df.drop(columns=['malignancy']).to_numpy(), np.array(list(map(lambda x: 1 if x==1 else 0, train_df['malignancy'].to_list()))).astype('float32')
        x_test, y_test = test_df.drop(columns=['malignancy']).to_numpy(), np.array(list(map(lambda x: 1 if x==1 else 0, test_df['malignancy'].to_list()))).astype('float32')
        
        history = compiled_model.fit(
            x_train, 
            y_train, 
            epochs=num_epochs,
            batch_size=x_train.shape[0],
            validation_data=(x_test,y_test)
        )
        y_pred = compiled_model.predict_classes(x_test, batch_size=x_test.shape[0])

        for metric_fn in metric_funcs:
            metrics_results[metric_fn.__name__].append(metric_fn(y_test, y_pred))        

    return metrics_results

###########################################################################

###########################################################################

def weighted_avg_and_std(values):
    average = np.average(values)
    variance = np.average((values-average)**2)
    return average, math.sqrt(variance)

# returns a dataframe with the mean and standard deviation from the results of a K-fold CV
def mean_std_results_k_fold_CV(k_fold_metrics_results: dict[str, list]):
    results_df = pd.DataFrame(columns=['metric', 'mean', 'std'])
    for metric_name, metric_results in k_fold_metrics_results.items():
        mean, std = weighted_avg_and_std(np.array(metric_results))
        results_df.append({
            'metric': metric_name,
            'mean': mean,
            'std': std
        })
    return results_df