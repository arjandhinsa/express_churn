

import pandas as pd


def avg_spend_per_visit(orders, visits, index_dates, k=1, prior=157.0):
    """
    Calculate the average spend per visit for each patient.

    Parameters:
    - orders: DataFrame containing order data with columns ['patient_key', 'order_date', 'value']
    - visits: DataFrame containing visit data with columns ['patient_key', 'visit_date', 'completed']
    - index_dates: DataFrame containing index dates with columns ['patient_key', 'index_date']
    - k: Smoothing parameter (default is 1)
    - prior: Prior value to add to the total spend (default is 157.0)

    Returns:
    - Series with 'patient_key' as index and a single column 'avg_spend_per_visit' containing the calculated average spend per visit for each patient.

    """
    # Merge visits with index_dates to filter visits before the index date
    merged_visits = pd.merge(visits, index_dates, on='patient_key')
    filtered_visits = merged_visits[merged_visits['visit_date'] <= merged_visits['index_date']]
    
    # Count completed visits per patient
    completed_visits = filtered_visits[filtered_visits['completed']]
    visit_counts = completed_visits.groupby('patient_key').size().reset_index(name='visit_count')
    
    # Merge orders with index_dates to filter orders before the index date
    merged_orders = pd.merge(orders, index_dates, on='patient_key')
    filtered_orders = merged_orders[merged_orders['order_date'] <= merged_orders['index_date']]
    
    # Sum order values per patient
    order_sums = filtered_orders.groupby('patient_key')['value'].sum().reset_index(name='total_value')
    
    # Merge visit counts and order sums
    result = pd.merge(visit_counts, order_sums, on='patient_key', how='outer').fillna(0)
    
    # Calculate average spend per visit
    result['avg_spend_per_visit'] = (result['total_value'] + prior) / (result['visit_count'] + k)
    
    return result.set_index("patient_key")["avg_spend_per_visit"]


def assign_tiers(values, cutoffs):
    """
    Assign tiers to patients based on their average spend per visit.
    
    Parameters:
    - values: Series containing average spend per visit for each patient.
    - cutoffs: List of cutoff values for tier assignment.
    
    Returns:
    - Series with the same index as 'values' containing the assigned tier for each patient.
    
    """
    labels = ["Bronze", "Silver", "Gold", "Platinum"]
    return pd.cut(values, bins=[-float("inf")] + list(cutoffs) + [float("inf")],
                  labels=labels, right=False)

