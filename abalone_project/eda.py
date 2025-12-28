import pandas as pd
import numpy as np

def prepare_training_data(df: pd.DataFrame):
    return pd.get_dummies(df, columns=['Sex'], prefix='Sex')

def transform_data(df: pd.DataFrame):
    result = pd.DataFrame()
    
    numeric_columns = ['Length', 'Diameter', 'Height', 
                      'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
    
    for col in numeric_columns:
        result[col] = df[col]
    
    result['Sex_F'] = (df['Sex'] == 'F').astype(int)
    result['Sex_I'] = (df['Sex'] == 'I').astype(int)
    result['Sex_M'] = (df['Sex'] == 'M').astype(int)
    
    return result
