def load_boston():
    import pandas as pd
    from sklearn.datasets import load_boston
    data = load_boston()
    X, y = pd.DataFrame(data['data'], columns = data.feature_names), pd.Series(data['target'], name='MEDV')
    return X, y
    