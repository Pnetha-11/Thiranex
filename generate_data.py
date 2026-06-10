import numpy as np
import pandas as pd
import os

def generate_messy_churn_data(num_samples=1200, seed=42):
    np.random.seed(seed)
    
    # 1. Customer IDs
    customer_ids = [f"CUST-{i:04d}" for i in range(1, num_samples + 1)]
    
    # 2. Demographic Features
    genders = np.random.choice(['Male', 'Female', 'MALE', 'female'], size=num_samples, p=[0.25, 0.25, 0.25, 0.25])
    senior_citizens = np.random.choice([0, 1], size=num_samples, p=[0.85, 0.15])
    partners = np.random.choice(['Yes', 'No', 'yes', 'no'], size=num_samples, p=[0.25, 0.25, 0.25, 0.25])
    dependents = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.3, 0.7])
    
    # 3. Account Features
    # Tenure (months) - normal range: 0 to 72 months. 
    tenures = np.random.randint(0, 73, size=num_samples)
    
    # Introduce negative tenure outliers (sensor/entry errors)
    neg_tenure_idx = np.random.choice(num_samples, size=15, replace=False)
    tenures[neg_tenure_idx] = -np.random.randint(1, 12, size=15)
    
    contracts = np.random.choice(['Month-to-month', 'One year', 'Two year'], size=num_samples, p=[0.55, 0.22, 0.23])
    paperless_billings = np.random.choice(['Yes', 'No'], size=num_samples, p=[0.6, 0.4])
    payment_methods = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'], 
        size=num_samples, p=[0.35, 0.25, 0.2, 0.2]
    )
    
    # 4. Service Features
    internet_services = np.random.choice(['DSL', 'Fiber optic', 'No'], size=num_samples, p=[0.3, 0.45, 0.25])
    
    online_security = []
    online_backup = []
    device_protection = []
    tech_support = []
    streaming_tv = []
    streaming_movies = []
    
    for i in range(num_samples):
        internet = internet_services[i]
        if internet == 'No':
            online_security.append('No internet service')
            online_backup.append('No internet service')
            device_protection.append('No internet service')
            tech_support.append('No internet service')
            streaming_tv.append('No internet service')
            streaming_movies.append('No internet service')
        else:
            online_security.append(np.random.choice(['Yes', 'No', 'yes', 'no'], p=[0.3, 0.3, 0.2, 0.2]))
            online_backup.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
            device_protection.append(np.random.choice(['Yes', 'No'], p=[0.4, 0.6]))
            tech_support.append(np.random.choice(['Yes', 'No', 'yes', 'no'], p=[0.3, 0.3, 0.2, 0.2]))
            streaming_tv.append(np.random.choice(['Yes', 'No'], p=[0.5, 0.5]))
            streaming_movies.append(np.random.choice(['Yes', 'No'], p=[0.5, 0.5]))
            
    # 5. Charges
    # Monthly Charges ($18 to $118 expected)
    monthly_charges = np.zeros(num_samples)
    for i in range(num_samples):
        internet = internet_services[i]
        if internet == 'No':
            monthly_charges[i] = np.random.uniform(18.0, 25.0)
        elif internet == 'DSL':
            monthly_charges[i] = np.random.uniform(45.0, 85.0)
        else: # Fiber optic
            monthly_charges[i] = np.random.uniform(70.0, 118.0)
            
    # Introduce monthly charge outliers (999.00 Tesla error and negative values)
    mag_outlier_idx = np.random.choice(num_samples, size=10, replace=False)
    monthly_charges[mag_outlier_idx] = 999.0
    neg_charge_idx = np.random.choice(num_samples, size=10, replace=False)
    monthly_charges[neg_charge_idx] = -np.random.uniform(10.0, 50.0, size=10)
    
    # Total Charges (roughly tenure * monthly_charges)
    total_charges = []
    for i in range(num_samples):
        t = max(0, tenures[i]) # Handle negative tenure logic
        m = monthly_charges[i]
        if m < 0 or m > 200: # Handle outliers
            m = 65.0
        tc = t * m + np.random.normal(0, 10)
        total_charges.append(max(0, tc))
    total_charges = np.array(total_charges)
    
    # Convert some Total Charges to NaNs / space strings (missing data)
    missing_charge_idx = np.random.choice(num_samples, size=35, replace=False)
    total_charges_str = [str(round(val, 2)) for val in total_charges]
    for idx in missing_charge_idx:
        if np.random.rand() > 0.5:
            total_charges_str[idx] = ' ' # Space string (Kaggle style null)
        else:
            total_charges_str[idx] = 'NaN' # Standard null
            
    # 6. Target Variable: Churn (Yes/No)
    # Built using a sigmoid logistic function based on risk parameters
    churn_prob = []
    for i in range(num_samples):
        # Base log-odds of churn
        log_odds = -1.2
        
        # Risk factors
        # 1. Contract type (Month-to-month is highly risky)
        contract = contracts[i]
        if contract == 'Month-to-month':
            log_odds += 1.8
        elif contract == 'One year':
            log_odds -= 0.5
        else:
            log_odds -= 1.5
            
        # 2. Tenure (Longer tenure means loyalty)
        t = tenures[i]
        if t > 0:
            log_odds -= (t / 15.0)
        else:
            log_odds += 0.8
            
        # 3. Internet service (Fiber optic has higher churn due to price sensitivity)
        internet = internet_services[i]
        if internet == 'Fiber optic':
            log_odds += 0.9
            
        # 4. Services (Online Security & Tech Support reduce churn)
        sec = online_security[i].lower()
        sup = tech_support[i].lower()
        if sec == 'yes':
            log_odds -= 0.6
        if sup == 'yes':
            log_odds -= 0.5
            
        # 5. Financial Factors (High Monthly Charges increase churn)
        m = monthly_charges[i]
        if 0 < m < 200:
            log_odds += (m - 60.0) / 40.0
            
        # Senior citizens are slightly more likely to churn
        if senior_citizens[i] == 1:
            log_odds += 0.4
            
        # Compute probability
        prob = 1.0 / (1.0 + np.exp(-log_odds))
        churn_prob.append(prob)
        
    churn_probs = np.array(churn_prob)
    churn = np.where(np.random.rand(num_samples) < churn_probs, 'Yes', 'No')
    
    # 7. Create DataFrame
    df = pd.DataFrame({
        'customer_id': customer_ids,
        'gender': genders,
        'senior_citizen': senior_citizens,
        'partner': partners,
        'dependents': dependents,
        'tenure': tenures,
        'phone_service': np.random.choice(['Yes', 'No'], size=num_samples, p=[0.9, 0.1]),
        'multiple_lines': np.random.choice(['Yes', 'No', 'No phone service'], size=num_samples, p=[0.4, 0.5, 0.1]),
        'internet_service': internet_services,
        'online_security': online_security,
        'online_backup': online_backup,
        'device_protection': device_protection,
        'tech_support': tech_support,
        'streaming_tv': streaming_tv,
        'streaming_movies': streaming_movies,
        'contract': contracts,
        'paperless_billing': paperless_billings,
        'payment_method': payment_methods,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges_str,
        'churn': churn
    })
    
    # Introduce duplicates (append 25 duplicate rows)
    dup_indices = np.random.choice(num_samples, size=25, replace=False)
    dup_df = df.iloc[dup_indices].copy()
    # Modify customer IDs slightly or keep them identical to represent duplicates
    df = pd.concat([df, dup_df], ignore_index=True)
    
    return df

if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    
    print("Generating raw, messy Customer Churn dataset...")
    df = generate_messy_churn_data(seed=101)
    
    output_path = os.path.join("data", "raw_churn_data.csv")
    df.to_csv(output_path, index=False)
    
    print(f"Success! Raw customer churn dataset saved to {output_path}")
    print(f"Dimensions: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Nulls injected in total_charges (represented as space/NaN): 35 records")
    print(f"Duplicates appended: 25 records")
    print(f"Monthly Charge outliers (negative / spike): 20 records")
    print(f"Tenure outliers (negative): 15 records")
