import xgboost as xgb
import numpy as np
import optuna
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
from functools import partial

X, y = make_classification(n_samples=10000, n_features=20, n_informative=6,
                           n_classes=6, n_clusters_per_class=4, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)


# Focal Loss Implementation for Multi-Class Classification
class FocalLoss:
    def __init__(self, gamma=2.0):
        self.gamma = gamma

    def focal_loss(self, predt, dtrain):
        """Compute the gradient and hessian for focal loss"""
        y = dtrain.get_label()
        p = np.exp(predt - np.max(predt, axis=1, keepdims=True))  # Softmax probabilities
        p /= np.sum(p, axis=1, keepdims=True)

        grad = np.zeros_like(predt)
        hess = np.zeros_like(predt)

        for i in range(predt.shape[1]):  # Loop over classes
            g = (p[:, i] - (y == i)) * ((1 - p[:, i]) ** self.gamma)
            h = (1 - 2 * p[:, i]) * g - self.gamma * (p[:, i] * np.log(p[:, i] + 1e-8)) * ((y == i) - p[:, i])
            grad[:, i] = g
            hess[:, i] = np.maximum(h, 1e-6)  # Avoid division by zero

        return grad.ravel(), hess.ravel()

    import numpy as np

    def softmax_xentropy(self, predt, dtrain):
        y = dtrain.get_label()  # Get the true labels from the training set
        n_classes = predt.shape[1]  # Number of classes (3 in this case)

        # Apply softmax to the predictions (if not already probabilities)
        exp_pred = np.exp(predt - np.max(predt, axis=1, keepdims=True))  # To avoid overflow
        p = exp_pred / np.sum(exp_pred, axis=1, keepdims=True)  # Softmax probabilities

        # Clip the predictions to avoid log(0) or NaN issues
        p = np.clip(p, 1e-7, 1 - 1e-7)

        # One-hot encode the true labels
        y_one_hot = np.eye(n_classes)[y.astype(int)]

        # Softmax Cross-Entropy Loss with gamma adjustment
        loss = -np.sum(y_one_hot * np.log(p) * ((1 - p) ** self.gamma), axis=1)

        return 'softmax_xentropy', np.mean(loss)  # Return the name and the loss value


# Optuna objective function for hyperparameter tuning
def objective(trial, X_train, y_train, X_valid, y_valid):
    # Suggest parameters for the model
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=10),  # Add n_estimators here
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True)

    }

    # Train the XGBoost model with the suggested hyperparameters
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)

    # Calculate validation score or any metric you'd like
    score = model.score(X_valid, y_valid)

    return score


# Run Optuna hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid), n_trials=5)

# Train final model with best hyperparameters
best_params = study.best_params
best_params["objective"] = "multi:softprob"
best_params["num_class"] = len(np.unique(y_train))

focal_loss = FocalLoss(gamma=2.0)
dtrain = xgb.DMatrix(X_train, label=y_train)
dvalid = xgb.DMatrix(X_valid, label=y_valid)

final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=best_params["n_estimators"],
    evals=[(dvalid, "validation")],
    obj=focal_loss.focal_loss,
    custom_metric=focal_loss.softmax_xentropy,
    early_stopping_rounds=20,
    verbose_eval=True,
)

# Evaluate on validation set
preds = final_model.predict(dvalid)
best_preds = np.argmax(preds, axis=1)
final_accuracy = accuracy_score(y_valid, best_preds)
print(f"Final Model Accuracy: {final_accuracy:.4f}")
