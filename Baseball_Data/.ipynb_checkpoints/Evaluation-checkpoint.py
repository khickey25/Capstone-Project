#create a pipeline score a particular model.
def eval_model(clf, X, y, cv):
    """Evaluate a model's performance using cross validation, using Random Oversampling
       to balance the classes of the target.
    Input: 
        clf, the classifier.
        X: data to be used in cross-validation
        y: corresponding values of the target
        cv: int, number of folds to use for cross-validation
    Output: average accuracy, f1 score, and AUC scores for number of folds ."""
    
    import numpy as np
    import imblearn.over_sampling 
    from imblearn.pipeline import make_pipeline
    from sklearn.model_selection import cross_validate
    
    #clf is Logistic Regression defined outside the function
    model = clf
    
    ros = imblearn.over_sampling.RandomOverSampler(ratio=1, random_state=777)
    pipe = make_pipeline(ros, model)
                       
    
    cv_results = cross_validate(pipe, X, y, scoring = ['accuracy', 'f1', 'roc_auc'], cv =cv,            return_estimator=True)

    
    
    for result in ['test_accuracy','train_accuracy','test_f1',
               'train_f1','test_roc_auc','train_roc_auc']:
        print(f"Mean {result} Value: {np.mean(cv_results[result])}")
        print(f"{result} scores: {cv_results[result]}")
        print()