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
    def __init__(self, gamma=2.0, alpha=1.0):
        self.gamma = gamma
        self.alpha = alpha  # Alpha to adjust the weight of classes

    def focal_loss(self, predt, dtrain):
        y = dtrain.get_label()
        p = np.exp(predt - np.max(predt, axis=1, keepdims=True))  # Softmax probabilities
        p /= np.sum(p, axis=1, keepdims=True)

        grad = np.zeros_like(predt)
        hess = np.zeros_like(predt)

        for i in range(predt.shape[1]):  # Loop over classes
            # Apply alpha to the class loss
            alpha_i = self.alpha if (y == i).any() else 1.0
            g = alpha_i * (p[:, i] - (y == i)) * ((1 - p[:, i]) ** self.gamma)
            h = alpha_i * (1 - 2 * p[:, i]) * g - self.gamma * (p[:, i] * np.log(p[:, i] + 1e-8)) * ((y == i) - p[:, i])
            grad[:, i] = g
            hess[:, i] = np.maximum(h, 1e-6)  # Avoid division by zero

        return grad, hess  # Return gradient and hessian

    def softmax_xentropy(self, predt, dtrain):
        y = dtrain.get_label()  # Get the true labels from the training set
        n_classes = predt.shape[1]  # Number of classes

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
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 1e-6, 1.0, log=True),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-6, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-6, 1.0, log=True),
        "objective": "multi:softprob",  # For multi-class classification
        "num_class": len(np.unique(y_train))  # Specify the number of classes
    }

    # Train the XGBoost model with the suggested hyperparameters
    model = xgb.XGBClassifier(**params, enable_categorical=True)
    model.fit(X_train, y_train)

    # Calculate validation score or any metric you'd like
    score = model.score(X_valid, y_valid)

    return score


# Run Optuna hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(partial(objective, X_train=X_train, y_train=y_train, X_valid=X_valid, y_valid=y_valid), n_trials=5)

# Train final model with best hyperparameters
best_params = study.best_params
best_params["num_class"] = len(np.unique(y_train))  # Add num_class here

# Custom Focal Loss
focal_loss = FocalLoss(gamma=2.0)

# Convert the datasets to DMatrix format
dtrain = xgb.DMatrix(X_train, label=y_train, enable_categorical=True)
dvalid = xgb.DMatrix(X_valid, label=y_valid, enable_categorical=True)

# Use a fixed value for num_boost_round
num_boost_round = 100  # This can be adjusted based on results from Optuna

# Final model training using xgb.train() with the custom loss and metric
final_model = xgb.train(
    best_params,
    dtrain,
    num_boost_round=num_boost_round,  # Use a fixed number of boosting rounds
    evals=[(dvalid, "validation")],
    obj=focal_loss.focal_loss,  # Use the custom focal loss function
    custom_metric=focal_loss.softmax_xentropy,  # Use the custom softmax cross-entropy metric
    early_stopping_rounds=20,
    verbose_eval=True,
)

# Get class probabilities
preds = final_model.predict(dvalid)  # Should return (n_samples, n_classes)
print("Shape of preds:", preds.shape)  # Debugging step

# Convert probabilities to class labels
if preds.ndim == 1:
    best_preds = preds  # If it's 1D, assume it contains class labels already
else:
    best_preds = np.argmax(preds, axis=1)  # Otherwise, get the highest probability class

# Compute final accuracy
final_accuracy = accuracy_score(y_valid, best_preds)
print(f"Final Model Accuracy: {final_accuracy:.4f}")

import numpy as np

# Get feature importance from the trained model
importance = final_model.get_score(importance_type='gain')

# Normalize the importance values
total_importance = sum(importance.values())
normalized_importance = {k: v / total_importance for k, v in importance.items()}

# Sort and print feature importance
sorted_importance = sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)

print("\nNormalized Feature Importance:")
for feature, score in sorted_importance:
    print(f"Feature {feature}: {score:.4f}")


