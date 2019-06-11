def preprocess_data(raw_data_file):
    '''Input the raw baseball file, 'Statcast_data.csv' as a pandas dataframe for data preprocessing.
       Go through preprocessing steps and output the data in a form suitable 
       to be trained on by machine learning models.
       Data formatting steps based on insights from EDA notebook.'''
    
    import pandas as pd
    import numpy as np
    import sklearn.preprocessing
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    
    bsb = pd.read_csv(raw_data_file, index_col = 0)
    
    #Begin data formatting; see EDA notebook
    #format the classes of the target, filter out to avoid transformations. 
    #Will store the target as a variable to
    #append back after all transformations taken place.
    bsb['description'] = bsb['description'].replace({'blocked_ball': 0, 'ball': 0, "called_strike": 1})
    target = bsb['description']
    bsb = bsb.drop(columns = 'description')
    
    #replace Knuckle Curve with Curveball
    bsb['pitch_name'] = bsb['pitch_name'].replace('Knuckle Curve', 'Curveball')
    
    
    #filter out dataframe to exclude any rows with Eephus
    bsb = bsb[bsb.pitch_name != 'Eephus']
            
    #Begin seperate preprocessing pipelines
    #seperate out categorical features;
    #these will be handled after numeric transformations
    #define numeric features for custom transformation:
    numeric_features = ['release_speed', 'release_spin_rate', 'release_pos_x',
       'release_pos_y', 'release_pos_z', 'pfx_x', 'pfx_z', 'vx0', 'vy0', 'vz0',
       'ax', 'ay', 'az', 'sz_top', 'sz_bot', 'release_extension']
    
    cat_features = ['pitch_name', 'player_name', 'p_throws']
    
    bsb_num = bsb.drop(columns = cat_features)
    
    #establish numeric pipeline steps; simple imputer and StandardScaler()
    num_pipe = Pipeline(steps = [
        ('impute', sklearn.preprocessing.Imputer(missing_values = np.nan, strategy = 'median')),
        ('scale', sklearn.preprocessing.StandardScaler())
                                ])
    #transform numeric data
    baseball = num_pipe.fit_transform(bsb_num)
    
    #make a dataframe to concat with categorical data that was filtered out
    baseball = pd.DataFrame(baseball, columns = bsb_num.columns)

    #concat the pitch_name and p_throws that was filtered out
    baseball['pitch_name'] = bsb['pitch_name']
    baseball['p_throws'] = bsb['p_throws']
    
    #get dummies, add a feature which indicates if the class value was missing or not
    baseball = pd.get_dummies(baseball, dummy_na=True)
    
    #add back the pitcher name
    baseball['player_name'] = bsb['player_name']
    
    #add back the target; note that there were no missing
    #values in the raw data, so no imputation was necessary
    baseball['description'] = target
    
    #however, there are 12 missing 'player_name' values in original data,
    #and since 'player_name' was not imputed,
    #we'll simply drop these 12 rows.
    baseball = baseball.dropna(how = 'any')
    
    return baseball