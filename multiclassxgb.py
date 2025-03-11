import xgboost as xgb
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.datasets import make_classification
from functools import partial

# ✅ Generate Synthetic Data
X, y = make_classification(n_samples=10000, n_features=20, n_informative=6,
                           n_classes=6, n_clusters_per_class=4, random_state=42)

# ✅ Convert to DataFrame (Ensures Focal Loss Compatibility)
X = pd.DataFrame(X)
y = pd.Series(y)

# ✅ Train-Test Split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# ✅ Custom Focal Loss for XGBoost
class FocalLoss:
    def __init__(self, gamma=2.0, alpha=1.0):
        self.gamma = gamma
        self.alpha = alpha

    def focal_loss(self, predt, dtrain):
        """ Compute gradient and hessian for Focal Loss. """
        y = dtrain.get_label()

        # Convert Pandas to NumPy if needed
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten()

        # Compute softmax probabilities
        exp_pred = np.exp(predt - np.max(predt, axis=1, keepdims=True))
        p = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        grad = np.zeros_like(predt)
        hess = np.zeros_like(predt)

        for i in range(predt.shape[1]):  # Loop over classes
            alpha_i = self.alpha if np.any(y == i) else 1.0  # Apply alpha per class
            p_t = p[:, i]
            y_t = (y == i).astype(float)

            # Compute gradient and hessian
            grad[:, i] = alpha_i * (p_t - y_t) * ((1 - p_t) ** self.gamma)
            hess[:, i] = np.maximum(
                alpha_i * (1 - 2 * p_t) * grad[:, i] - self.gamma * p_t * np.log(p_t + 1e-8) * (y_t - p_t),
                1e-6,  # Avoid division by zero
            )

        return grad, hess

    def softmax_xentropy(self, predt, dtrain):
        """ Compute softmax cross-entropy with gamma adjustment. """
        y = dtrain.get_label()

        # Convert Pandas to NumPy if needed
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.flatten()

        n_classes = predt.shape[1]

        # Compute softmax probabilities
        exp_pred = np.exp(predt - np.max(predt, axis=1, keepdims=True))
        p = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)

        # Clip probabilities to avoid log(0)
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # One-hot encoding of true labels
        y_one_hot = np.eye(n_classes)[y.astype(int)]

        # Compute loss
        loss = -np.sum(y_one_hot * np.log(p) * ((1 - p) ** self.gamma), axis=1)

        return 'softmax_xentropy', np.mean(loss)

# ✅ Convert Data to DMatrix (for xgb.train)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

# ✅ Define Hyperparameter Optimization Function
def objective(trial):
    """ Optimize XGBoost Hyperparameters with Optuna """

    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        "alpha": trial.suggest_float("alpha", 0.1, 2.0),  # Added alpha as a hyperparameter
        "objective": "multi:softprob",  # Multi-class classification
        "num_class": len(np.unique(y_train))  # Specify number of classes
    }

    # ✅ Train XGB Model using Custom Focal Loss
    focal_loss = FocalLoss(gamma=2.0, alpha=1.0)
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=1000,
        evals=[(dvalid, "validation")],
        obj=focal_loss.focal_loss,  # Use Custom Focal Loss
        custom_metric=focal_loss.softmax_xentropy,  # Custom Evaluation Metric
        early_stopping_rounds=20,
        verbose_eval=False
    )

    # ✅ Get Predictions
    preds = model.predict(dvalid)
    best_preds = np.argmax(preds, axis=1)  # Convert probabilities to class labels

    # ✅ Compute Weighted F1-score (Handles Imbalanced Data)
    return f1_score(y_valid, best_preds, average="weighted")

# ✅ Run Optuna Optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=5)

# ✅ Train Final Model with Best Hyperparameters
best_params = study.best_params
best_params["num_class"] = len(np.unique(y_train))  # Ensure num_class is set

# ✅ Train Final Model using XGBoost with Focal Loss
focal_loss = FocalLoss(gamma=2.0, alpha=1.0)
final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=1000,
    evals=[(dvalid, "validation")],
    obj=focal_loss.focal_loss,  # Use Custom Focal Loss
    custom_metric=focal_loss.softmax_xentropy,  # Custom Metric
    early_stopping_rounds=20,
    verbose_eval=True
)

# ✅ Get Final Predictions
best_preds = final_model.predict(dvalid)
# best_preds = np.argmax(preds, axis=1)  # Convert probabilities to class labels

# ✅ Compute Final Accuracy and Weighted F1-score
final_accuracy = accuracy_score(y_valid, best_preds)
final_f1 = f1_score(y_valid, best_preds, average="weighted")  # Weighted F1-score

print(f"Final Model Accuracy: {final_accuracy:.4f}")
print(f"Final Model Weighted F1-Score: {final_f1:.4f}")
