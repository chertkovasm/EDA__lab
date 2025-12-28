import pandas as pd
import joblib
from explainerdashboard import RegressionExplainer, ExplainerDashboard
from sklearn.preprocessing import RobustScaler
from ucimlrepo import fetch_ucirepo

from eda import prepare_training_data

def generate_dashboard():
    model = joblib.load("/app/trained_model/trained_model.pkl")
    scaler = joblib.load("/app/trained_model/scaler.pkl")
    
    abalone = fetch_ucirepo(id=1)
    X = abalone.data.features
    y = abalone.data.targets
    
    
    df = X.copy()
    df["Rings"] = y

    df = prepare_training_data(df)
    x = df.drop(["Rings"], axis=1)
    y_data = df["Rings"]
    
    X_scaled = scaler.transform(x)
    X_scaled_df = pd.DataFrame(X_scaled, columns=x.columns)

    explainer = RegressionExplainer(model, X_scaled_df[:100], y_data[:100])
    db = ExplainerDashboard(explainer, title="Abalone Age Prediction")
    db.to_yaml("dashboard.yaml", explainerfile="explainer.joblib", dump_explainer=True)
    
    print("Дашборд создан")

if __name__ == '__main__':
    generate_dashboard()