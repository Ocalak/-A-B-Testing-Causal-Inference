import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

def propensity_score_matching(data):
    df = data.copy()
    df['group_binary'] = (df['group'] == 'treatment').astype(int)
    
    # Fit logistic regression for propensity scores
    model = LogisticRegression()
    features = ['age', 'income', 'visit_frequency']
    model.fit(df[features], df['group_binary'])
    
    df['propensity_score'] = model.predict_proba(df[features])[:,1]
    
    return df

def inverse_probability_weighting(df):
    treated = df['group_binary'] == 1
    control = df['group_binary'] == 0
    
    # Calculate weights
    df['weight'] = np.where(
        treated,
        1/df['propensity_score'],
        1/(1 - df['propensity_score'])
    )
    
    # Weighted average treatment effect
    treated_mean = np.average(df.loc[treated, 'converted'], weights=df.loc[treated, 'weight'])
    control_mean = np.average(df.loc[control, 'converted'], weights=df.loc[control, 'weight'])
    
    ate = treated_mean - control_mean
    return ate
