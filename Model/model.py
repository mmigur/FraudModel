import joblib
import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

with open('../Configs/config.yaml', 'r') as config_file:
    CONFIG = yaml.load(config_file, Loader=yaml.FullLoader)

def train():
    def get_data() -> pd.DataFrame:
        fraud_df = pd.read_csv(CONFIG['PATH_TO_DATA'])
        fraud_df['Class']= fraud_df['Class'].astype('category')
        fraud_df[['Time', 'Amount']] = StandardScaler().fit_transform(fraud_df[['Time', 'Amount']])
        return fraud_df
    
    fraud_df = get_data()

    X = fraud_df.drop('Class', axis=1)
    y = fraud_df['Class']

    model_MLP = MLPClassifier(
        solver=CONFIG['SOLVER'], 
        activation=CONFIG['ACTIVATION'], 
        hidden_layer_sizes=(CONFIG['HIDDEN_LAYER_SIZES'],),
        learning_rate=CONFIG['LEARNING_RATE'], 
        max_iter=CONFIG['MAX_ITER'], 
        random_state=CONFIG['RANDOM_STATE']
    ).fit(X, y)

    joblib.dump(model_MLP, CONFIG['PATH_TO_SAVE'])

def predict(object: list) -> int:
    model = joblib.load(CONFIG['PATH_TO_MODEL'])
    return model.predict(object).tolist()[0]