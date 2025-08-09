def train_xgboost_with_optuna(df, target_column):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    def objective(trial):
        param = {
            "verbosity": 0,
            "objective": "reg:squarederror",
            "tree_method": "auto",
            "booster": "gbtree",
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.3, 1.0),
            "subsample": trial.suggest_float("subsample", 0.3, 1.0),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        }
        model = xgb.XGBRegressor(**param)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return mean_squared_error(y_test, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30,show_progress_bar=True)

    best_params = study.best_trial.params
    print("Best hyperparameters:", best_params)

    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("R² Score:", r2_score(y_test, preds))
    print("MSE:", mean_squared_error(y_test, preds))

    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.summary_plot(shap_values, X_test, plot_type="bar")
    plt.show()

    return model

if __name__ =="__main__":
  train_xgboost_with_optuna(df,target_column='energy_per_atom')
  plt.show()
