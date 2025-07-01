import pandas as pd
import numpy as np

def generate_customer_data(n_customers=5000, seed=42):
    np.random.seed(seed)
    data = pd.DataFrame({
        'customer_id': np.arange(n_customers),
        'age': np.random.randint(18, 65, size=n_customers),
        'income': np.random.normal(60000, 15000, n_customers).astype(int),
        'visit_frequency': np.random.poisson(3, n_customers),
    })
    data['group'] = np.random.choice(['control', 'treatment'], size=n_customers)
    
    # Baseline conversion probability
    base_prob = 0.1 + 0.002 * (data['visit_frequency']) + 0.000005 * (data['income'])

    # Treatment effect
    treatment_effect = 0.05  # 5% uplift
    data['conversion_prob'] = base_prob + (data['group'] == 'treatment') * treatment_effect
    
    # Simulate conversion
    data['converted'] = np.random.binomial(1, data['conversion_prob'])
    
    return data

if __name__ == "__main__":
    df = generate_customer_data()
    df.to_csv('data/simulated_customer_data.csv', index=False)
    print("Data saved to data/simulated_customer_data.csv")
