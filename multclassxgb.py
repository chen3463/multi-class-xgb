import numpy as np
import pandas as pd
import xgboost as xgb
import shap
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score
from sklearn.inspection import permutation_importance
from scipy.special import softmax
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Sample Data (Replace with actual dataset)
X = pd.DataFrame(np.random.randn(10000, 20), columns=[f'feat_{i}' for i in range(20)])
y = np.random.randint(0, 4, 10000)  # Multi-class labels (4 classes)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scale Features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---- 1. Feature Selection (SHAP + Permutation Importance) ----
def feature_selection(model, X_train, y_train):
    """Select features using SHAP and Permutation Importance."""
    model.fit(X_train, y_train)
    
    # SHAP Analysis
    explainer = shap.Explainer(model)
    shap_values = explainer(X_train)
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    
    # Permutation Importance
    perm_importance = permutation_importance(model, X_train, y_train, scoring='f1_weighted', n_repeats=10)
    
    # Aggregate Feature Scores
    importance = shap_importance + perm_importance.importances_mean
    selected_features = np.argsort(importance)[-15:]  # Select top 15 features

    return selected_features

# ---- 2. Custom Focal Loss for Multi-Class ----
def focal_loss(predt, dtrain, gamma=2.0):
    """Compute Focal Loss for XGBoost."""
    labels = dtrain.get_label().astype(int)
    num_classes = predt.shape[1]
    pred_probs = softmax(predt, axis=1)

    pt = pred_probs[np.arange(len(labels)), labels]
    grad = (1 - pt) ** gamma * (pt - 1)[:, None] * (labels[:, None] == np.arange(num_classes))
    hess = (1 - pt) ** gamma * (pt * (1 - pt) + gamma * (pt - 1) * np.log(pt))[:, None] * (labels[:, None] == np.arange(num_classes))

    return grad.flatten(), hess.flatten()

# ---- 3. Custom AUCPR Metric ----
def aucpr_metric(y_true, y_pred):
    """Compute AUCPR for Multi-Class Classification."""
    pred_probs = softmax(y_pred, axis=1)
    aucpr_scores = [average_precision_score((y_true == c).astype(int), pred_probs[:, c]) for c in range(pred_probs.shape[1])]
    return np.mean(aucpr_scores)

# ---- 4. Hyperparameter Tuning using Optuna ----
def objective(trial):
    """Optimize XGBoost hyperparameters."""
    params = {
        "learning_rate": trial.suggest_loguniform("learning_rate", 0.01, 0.3),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "gamma": trial.suggest_loguniform("gamma", 0.5, 5.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_uniform("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_uniform("colsample_bytree", 0.6, 1.0),
        "num_class": 4,
        "objective": "multi:softprob"
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train[:, selected_features], y_train)
    
    y_pred = model.predict_proba(X_test[:, selected_features], output_margin=True)
    return aucpr_metric(y_test, y_pred)

# ---- 5. Run Hyperparameter Tuning ----
base_model = xgb.XGBClassifier(objective="multi:softprob", num_class=4)
selected_features = feature_selection(base_model, X_train, y_train)

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

best_params = study.best_params
print("Best Parameters:", best_params)

# ---- 6. Train Final Model with Best Hyperparameters ----
final_model = xgb.XGBClassifier(**best_params)
final_model.fit(X_train[:, selected_features], y_train)

# ---- 7. Evaluate Final Model ----
y_pred = final_model.predict_proba(X_test[:, selected_features], output_margin=True)
final_aucpr = aucpr_metric(y_test, y_pred)
print(f"Final AUCPR: {final_aucpr:.4f}")
