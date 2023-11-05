# imports
import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
import numpy as np
import math
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA

# setting a numpy seed
np.random.seed(42)


###########################################################################

def get_k_folds(df, k=10, seed=None):

    """
    get_k_folds: Split a DataFrame into k folds for cross-validation, ensuring balanced class proportions 
    and ensuring all annotations from a certain patient are kept in the same fold.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be split into folds.
    - k (int, optional): The number of folds (default is 10).
    - seed (int, optional): Random seed for reproducibility (default is None).

    Returns:
    - List of DataFrames: A list containing k folds of the input DataFrame.

    This function takes a DataFrame and splits it into k folds for cross-validation while ensuring that the 
    proportion of samples in each class (malignancy) is balanced across all folds, and ensuring no sample 
    from the same patient is in a different fold. It does this by shuffling the data and assigning patients 
    to folds, ensuring that each fold has approximately the same distribution of malignancy classes.

    Notes:
    - If it's not possible to achieve balanced class proportions, the function will reattempt the split until 
    it succeeds.
    - The 'patient_id' column is removed from the resulting folds as it is a categorical, not decisive 
    for the target feature.
    """

    def valid_proportions(folds):
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
        shuffled_pid_list = pd.Series(pid_list).sample(frac=1, random_state=seed).to_list()
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
    
    for i in range(len(folds)):
        folds[i].drop(columns=['patient_id'], inplace=True)

    return folds

###########################################################################

def k_fold_cv(model, df, k=10, metric_funcs=[f1_score, accuracy_score, roc_auc_score], k_fold_verbose=False, pca_components=0, show_confusion_matrix=False, seed=1):
    
    """
    k_fold_cv: Perform k-fold cross-validation for a machine learning model with various evaluation metrics.

    Parameters:
    - model: An instance of a machine learning model that has .fit() and .predict() methods.
    - df (pd.DataFrame): The DataFrame containing the dataset to be used for cross-validation.
    - k (int, optional): The number of folds for cross-validation (default is 10).
    - metric_funcs (list, optional): List of evaluation metrics functions to apply (default includes F1 score, 
    accuracy, and ROC AUC score).
    - k_fold_verbose (bool, optional): Whether to print verbose information during the k-fold process (default 
    is False).
    - pca_components (int, optional): The number of PCA components to use for dimensionality reduction (default 
    is 0, no PCA).
    - show_confusion_matrix (bool, optional): Whether to display an average confusion matrix (default is False).
    - seed (int, optional): Random seed for reproducibility (default is 1).

    Returns:
    - Dictionary: A dictionary containing evaluation metrics as keys and lists of results for each fold as values.

    This function performs k-fold cross-validation for a machine learning model using the specified evaluation 
    metrics. It divides the input DataFrame into k folds, trains the model on k-1 folds, and evaluates it on 
    the remaining fold. The evaluation metrics are calculated for each fold, and the results are stored in a 
    dictionary.

    Notes:
    - If show_confusion_matrix is set to True, a confusion matrix constituted by the average values of the confusion 
    matrices that would be produced in each fold will be displayed for the folds.
    - Considering there is a default seed set to 1, the folds received and used for cross-validation will always 
    be deterministic. This can be changed by setting the seed to None.
    """
    
    folds = get_k_folds(df, k, seed)
    
    metrics_results = dict((metric_fn.__name__, []) for metric_fn in metric_funcs)

    first_verbose = k_fold_verbose
    cm_list = []
    for test_fold_index in range(len(folds)):
        if first_verbose:
            print("Performing K-Fold CV:", end="")
            first_verbose = False
        if k_fold_verbose:
            print(f" {test_fold_index+1}", end="")

        testing_df = folds[test_fold_index]
        training_df = pd.DataFrame(columns=folds[0].columns)
        for fold_index, fold_df in enumerate(folds):
            if fold_index == test_fold_index:
                continue
            training_df = pd.concat([training_df, fold_df], ignore_index=True)

        label_encoder = LabelEncoder()
        X_train, y_train = training_df.drop(columns=['malignancy']), label_encoder.fit_transform(training_df['malignancy'])
        X_test, y_test = testing_df.drop(columns=['malignancy']), label_encoder.fit_transform(testing_df['malignancy'])

        if pca_components > 0:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.fit_transform(X_test)
            pca = PCA(n_components=pca_components, random_state=42)
            X_train = pca.fit_transform(X_train)
            X_test = pca.fit_transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if show_confusion_matrix:
            cm_list.append(confusion_matrix(y_test, y_pred))
            

        for metric_fn in metric_funcs:
            metrics_results[metric_fn.__name__].append(metric_fn(y_test, y_pred))
    
    if show_confusion_matrix:
        mean_cm = np.array([[0,0],
                            [0,0]])
        total = 0
        for cm in cm_list:
            for i in range(2):
                for j in range(2):
                    mean_cm[i][j] += cm[i][j]
                    total += cm[i][j]
        mean_cm = mean_cm / total * 100
        mean_cm = mean_cm.round(decimals=0)
        disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm)
        disp.plot()

    if k_fold_verbose:
        print()
    return metrics_results

###########################################################################

def k_fold_cv_keras(compiled_model, df, k=10, metric_funcs=[f1_score, accuracy_score, roc_auc_score], num_epochs=10, k_fold_verbose=False, keras_verbose=0, pca_components=0, show_confusion_matrix=False, seed=1):
    
    """
    k_fold_cv_keras: Perform k-fold cross-validation for a TensorFlow Keras model with various evaluation metrics.

    Parameters:
    - compiled_model: A compiled TensorFlow Keras model.
    - df (pd.DataFrame): The DataFrame containing the dataset to be used for cross-validation.
    - k (int, optional): The number of folds for cross-validation (default is 10).
    - metric_funcs (list, optional): List of evaluation metrics functions to apply (default includes F1 score, 
    accuracy, and ROC AUC score).
    - num_epochs (int, optional): The number of training epochs for each fold (default is 10).
    - k_fold_verbose (bool, optional): Whether to print verbose information during the k-fold process (default 
    is False).
    - keras_verbose (int, optional): Keras verbose level for model training (default is 0, training progress not 
    shown).
    - pca_components (int, optional): The number of PCA components to use for dimensionality reduction (default 
    is 0, no PCA).
    - show_confusion_matrix (bool, optional): Whether to display an average confusion matrix (default is False).
    - seed (int, optional): Random seed for reproducibility (default is 1).

    Returns:
    - Dictionary: A dictionary containing evaluation metrics as keys and lists of results for each fold as values.

    This function performs k-fold cross-validation for a TensorFlow Keras model using the specified evaluation 
    metrics. It divides the input DataFrame into k folds, trains the model on k-1 folds, and evaluates it on the 
    remaining fold. The evaluation metrics are calculated for each fold, and the results are stored in a dictionary.

    Notes:   
    - If show_confusion_matrix is set to True, a confusion matrix constituted by the average values of the confusion 
    matrices that would be produced in each fold will be displayed for the folds.
    - Considering there is a default seed set to 1, the folds received and used for cross-validation will always 
    be deterministic. This can be changed by setting the seed to None.
    - The model must use the softmax activation function on the output layer.
    """
    
    folds = get_k_folds(df, k, seed)

    metrics_results = dict((metric_fn.__name__, []) for metric_fn in metric_funcs)

    first_verbose = k_fold_verbose
    cm_list = []
    for index, fold in enumerate(folds):
        if first_verbose:
            print("Performing K-Fold CV:", end="")
            first_verbose = False
        if k_fold_verbose:
            print(f" {index+1}", end="")

        test_df = fold
        train_df = pd.DataFrame(columns=fold.columns)
        for i in range(len(folds)):
            if i == index: continue
            train_df = pd.concat([train_df, fold], ignore_index=True)
        
        x_train, y_train = train_df.drop(columns=['malignancy']).to_numpy(dtype=np.float32), np.array(list(map(lambda x: 1 if x==1 else 0, train_df['malignancy'].to_list()))).astype('float32')
        x_test, y_test = test_df.drop(columns=['malignancy']).to_numpy(dtype=np.float32), np.array(list(map(lambda x: 1 if x==1 else 0, test_df['malignancy'].to_list()))).astype('float32')
        
        if pca_components > 0:
            scaler = StandardScaler()
            x_train = scaler.fit_transform(x_train)
            x_test = scaler.fit_transform(x_test)
            pca = PCA(n_components=pca_components, random_state=42)
            x_train = pca.fit_transform(x_train)
            x_test = pca.fit_transform(x_test)

        history = compiled_model.fit(
            x_train, 
            y_train, 
            epochs=num_epochs,
            batch_size=x_train.shape[0],
            verbose=keras_verbose
        )
        y_pred = np.argmax(compiled_model.predict(x_test, batch_size=x_test.shape[0]), axis=-1) #compiled_model.predict_classes(x_test, batch_size=x_test.shape[0])

        if show_confusion_matrix:
            cm_list.append(confusion_matrix(y_test, y_pred))
            
        for metric_fn in metric_funcs:
            metrics_results[metric_fn.__name__].append(metric_fn(y_test, y_pred))

    if show_confusion_matrix:
        mean_cm = np.array([[0,0],
                            [0,0]])
        total = 0
        for cm in cm_list:
            for i in range(2):
                for j in range(2):
                    mean_cm[i][j] += cm[i][j]
                    total += cm[i][j]
        mean_cm = mean_cm / total * 100
        mean_cm = mean_cm.round(decimals=0)
        disp = ConfusionMatrixDisplay(confusion_matrix=mean_cm)
        disp.plot()

    if k_fold_verbose:
        print()
    return metrics_results

###########################################################################

def avg_and_std(values):

    """
    avg_and_std: Calculate the average and standard deviation of a list of values.

    Parameters:
    - values (numpy array): A list of numeric values.

    Returns:
    - Float: the average of the input values.
    - Float: the standard deviation of the input values.

    This function takes a list of numeric values and computes their average and standard deviation.
    """

    average = np.average(values)
    variance = np.average((values-average)**2)
    return average, math.sqrt(variance)

###########################################################################

def mean_std_results_k_fold_CV(k_fold_metrics_results):

    """
    mean_std_results_k_fold_CV: Calculate the mean and standard deviation of evaluation metrics from K-fold 
    cross-validation results.

    Parameters:
    - k_fold_metrics_results (dictionary): A dictionary containing evaluation metrics and their results from 
    K-fold cross-validation.

    Returns:
    - pd.DataFrame: A DataFrame with metric names, mean values, and standard deviations.

    This function takes a dictionary of evaluation metrics and their results obtained from K-fold 
    cross-validation and calculates the mean and standard deviation of each metric's results and returns 
    them in a DataFrame.
    """

    metrics_list = []
    for metric_name, metric_results in k_fold_metrics_results.items():
        mean, std = avg_and_std(np.array(metric_results))
        metrics_list.append({
            'metric': metric_name,
            'mean': mean,
            'std': std
        })
    results_df = pd.DataFrame(data=metrics_list, columns=['metric', 'mean', 'std'])
    return results_df