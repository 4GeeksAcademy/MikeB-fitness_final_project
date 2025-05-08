import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV


def split_data(df, test_size=0.2, random_state=42):
    """Splits the dataset into train_df and test_df."""
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    
    return train_df, test_df 

def tune_hyperparameters(model, X_train, y_train, param_grid, cv=3, n_jobs=-1):
    """
    Performs hyperparameter tuning using GridSearchCV.

    Args:
        model: Machine learning model instance.
        X_train (pd.DataFrame or np.array): Training feature dataset.
        y_train (pd.Series or np.array): Training labels.
        param_grid (dict): Dictionary of hyperparameters to search.
        cv (int): Number of cross-validation folds (default: 3).
        n_jobs (int): Number of parallel jobs (-1 uses all available cores).

    Returns:
        tuple: Best estimator model and best hyperparameters dictionary.
    """
    search = GridSearchCV(model, param_grid, return_train_score=True, cv=cv, n_jobs=n_jobs)
    search_results = search.fit(X_train, y_train)
    
    best_model = search_results.best_estimator_
    best_params = search_results.best_params_

    print("Best hyperparameters:\n")
    for key, value in best_params.items():
        print(f" {key}: {value}")

    return best_model, best_params

def get_boosted_feature_importance(model, X_test, y_test, feature_names):
    """Uses permutation importance for HistGradientBoostingRegressor."""
    perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": perm_importance.importances_mean
    }).sort_values(by="Importance", ascending=False)

    return importance_df

def get_feature_importance(model, feature_names):
    """Retrieves and sorts feature importance scores."""
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values(by="Importance", ascending=False)

    return importance_df

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates RMSE and prints results."""
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred)
    print(f"{model_name} RMSE: {rmse:.2f}")
    return rmse

def evaluate_r2(model, X_test, y_test, model_name):
    """Calculates R² score to assess fit."""
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"{model_name} R² Score: {r2:.2f}")
    return r2
