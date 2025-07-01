import pandas as pd
from scipy import stats
import numpy as np

def ab_test_summary(data):
    control = data[data['group'] == 'control']
    treatment = data[data['group'] == 'treatment']
    
    conv_control = control['converted'].mean()
    conv_treatment = treatment['converted'].mean()
    
    lift = conv_treatment - conv_control
    
    # Two-proportion z-test
    n_control = control.shape[0]
    n_treatment = treatment.shape[0]
    successes = [control['converted'].sum(), treatment['converted'].sum()]
    samples = [n_control, n_treatment]
    
    stat, pval = stats.proportions_ztest(successes, samples)
    
    return {
        'conversion_control': conv_control,
        'conversion_treatment': conv_treatment,
        'lift': lift,
        'z_stat': stat,
        'p_value': pval
    }

def bootstrap_difference(data, n_bootstrap=10000):
    control = data[data['group'] == 'control']['converted']
    treatment = data[data['group'] == 'treatment']['converted']
    diffs = []

    for _ in range(n_bootstrap):
        boot_control = np.random.choice(control, size=len(control), replace=True)
        boot_treatment = np.random.choice(treatment, size=len(treatment), replace=True)
        diffs.append(boot_treatment.mean() - boot_control.mean())
    
    ci = np.percentile(diffs, [2.5, 97.5])
    return ci, diffs
