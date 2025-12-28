import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import RobustScaler
from ucimlrepo import fetch_ucirepo

from eda import prepare_training_data

if __name__ == '__main__':

    abalone = fetch_ucirepo(id=1)
    X = abalone.data.features
    y = abalone.data.targets

    df = X.copy()
    df["Rings"] = y

    df = prepare_training_data(df)
    x = df.drop(["Rings"], axis=1)
    y = df["Rings"]

    y_bins = np.zeros(len(y))
    y_bins[y < 8] = 0
    y_bins[(y >= 8) & (y < 10)] = 1
    y_bins[y >= 10] = 2

    scaler = RobustScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=43, stratify=y_bins
    )

    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    rf = RandomForestRegressor(random_state=43)
    rf_search = GridSearchCV(
        rf, rf_param_grid, cv=3,
        scoring='neg_mean_squared_error', n_jobs=-1
    )
    rf_search.fit(x_train, y_train)

    best_model = rf_search.best_estimator_

    joblib.dump(best_model, '/app/trained_model/trained_model.pkl')
    joblib.dump(scaler, '/app/trained_model/scaler.pkl')

    print("Модель сохранена")