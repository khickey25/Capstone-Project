#create a pipeline score a particular model.
def eval_model(clf, X, y, cv):
    """Evaluate a model's performance using cross validation.
    Input: 
        clf, the classifier.
        X: data to be used in cross-validation
        y: corresponding values of the target
        cv: int, number of folds to use for cross-validation
    Output: average accuracy, f1 score, and AUC scores for number of folds ."""
    
    import numpy as np
    from sklearn.model_selection import cross_validate
    
    #clf is Logistic Regression defined outside the function            
    
    cv_results = cross_validate(clf, X, y, scoring = ['accuracy', 'f1', 'roc_auc'], 
                                cv =cv, return_estimator=True, return_train_score = True)

    for result in ['train_accuracy', 'test_accuracy', 'train_f1', 'test_f1', 'train_roc_auc', 'test_roc_auc']:
        print(f"Mean {result} Value: {np.mean(cv_results[result])}")
        print(f"{result} scores: {cv_results[result]}")
        print() 
    

        
def hyperparam_search_pipeline(clf, X, y, parameter_dict, n_iter, cv, refit):
    '''Build Random Search Pipeline using defined classifier with Random Oversampler Strategy
    clf is classifer want to use
    X is data used to train on
    y is the target data corresponding to X
    params is the hyperparamater space want to search through
    n_iter is number of random search iterations
    cv is number of cross-validation
    refit ('Accuracy', 'AUC', 'F1') is string of which scoring metric want to display for results
    Returns: cv_results dictionary, best_estimator object'''
    
    #Define the Pipeline, the parameters, the scoring metrics, and finally
    #the ransom_search object that uses the pipeline
    from imblearn.pipeline import Pipeline
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.model_selection import RandomizedSearchCV

                   
    #define hyperparameters for classifier
    params = parameter_dict

    #define the three scoring metrics
    scoring = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'F1': 'f1'}

    #create Random Search Object, using the pipe object
    random_search = RandomizedSearchCV(clf, param_distributions=params,
                                n_iter=n_iter, n_jobs=-1, scoring = scoring, refit = refit,
                                return_train_score=True, cv = cv, random_state= 777)
    #Fit to the training splits
    random_search.fit(X, y)
    results_rs = random_search.cv_results_
    
    print(f"Best performing model is: {random_search.best_estimator_[1]}")
    
    return random_search.best_estimator_, results_rs

def plot_search_results(cv_results, param,  metrics):
    """cv_results of random_search.fit(),
       param is name of parameter want to graph on x-axis, will be in form of 'param_clf_{metric name}'
       metrics is a dict of scoring metrics
       metrics = {'AUC': 'roc_auc', 'Accuracy': 'accuracy', 'F1': 'f1'}
       """
    import numpy as np
    import matplotlib.pyplot as plt

    plt.figure(figsize=(13, 13))
    plt.title("Hyperparameter search results",
          fontsize=16)

    plt.xlabel("max_depth")
    plt.ylabel("Score")

    ax = plt.gca()
    ax.set_xlim(cv_results[str(param)].data.min(),(cv_results[str(param)].data.max()))
    ax.set_ylim(0.4, 1)

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(cv_results[str(param)].data, dtype=float)

    for scorer, color in zip(sorted(metrics), ['g', 'k', 'r']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = cv_results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = cv_results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(cv_results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = cv_results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
            linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    plt.show()
    
def plot_validation_curve(clf, X, y, param_range, param_name, metric):

    import matplotlib.pyplot as plt
    import numpy as np

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import validation_curve

    #param_range = np.linspace(1, 100, 5)
    train_scores, test_scores = validation_curve(
        clf, X, y, param_name=param_name, param_range=param_range,
        cv=5, scoring=metric, n_jobs=-1)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve with Decision Tree")
    plt.xlabel(f"{param_name}")
    plt.ylabel(f" {metric} Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.plot(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.plot(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt.show()
    
    
def eval_test_set(x_test_predictions, y_test):
    
    from sklearn import metrics
    print(f"Accuracy Score: {metrics.accuracy_score(y_test, x_test_predictions)}")
    print()
    print(f"AUC Score: {metrics.roc_auc_score(y_test, x_test_predictions)}")
    print()
    print(f"F1 Score: {metrics.f1_score(y_test, x_test_predictions)}")
    print()
    print(f"Classification Report: \n {metrics.classification_report(y_test, x_test_predictions)}")
    print()
    print(f" Confustion Matrix: \n {metrics.confusion_matrix(y_test, x_test_predictions)}")
    
def score_regressor(data, target, regressor):
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    import numpy as np
    test_mse_scores = []
    test_r2_scores = []
    for random_state in range(1000):
        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size = .25, random_state = random_state)
    
        regressor.fit(X_train, y_train)
        y_pred = regressor.predict(X_test)
    
        mse = mean_squared_error(y_test, y_pred)
        test_mse_scores.append(mse)
        
        r2 = r2_score(y_test, y_pred)
        test_r2_scores.append(r2)


    rmse_scores = [np.sqrt(score) for score in test_mse_scores]
    print(f" Average Test RMSE: {np.mean(rmse_scores)}")
    print()
    print(f" Average Test R^2 Score: {np.mean(test_r2_scores)}")
    print()
    print(f"First 30 RMSE scores: {np.round(rmse_scores[:30],4)}")
    print()
    print(f"First 30 R^2 scores: {np.round(test_r2_scores[:30], 3)}")
    
    return rmse_scores, test_r2_scores