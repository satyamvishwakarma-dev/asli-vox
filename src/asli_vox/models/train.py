import lightgbm as lgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_model(X, y):
    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Define parameters
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1
    }

    # Create Datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_valid, label=y_valid, reference=train_data)

    # Train the model
    model = lgb.train(
        params,
        train_data,
        num_boost_round=100,
        valid_sets=[train_data, valid_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=10)
        ]
    )

# Save the model
    model.save_model('lightgbm_model.txt')

# Sanity check
    y_pred = model.predict(X_valid, num_iteration=model.best_iteration)
    # Normalize LightGBM's typed prediction result before thresholding it.
    y_pred_class = (np.asarray(y_pred, dtype=np.float64) > 0.5).astype(int)

# Print classification report and confusion matrix
    print(classification_report(y_valid, y_pred_class))
    print(confusion_matrix(y_valid, y_pred_class))

# Check if the model is predicting only one class
    if len(set(y_pred_class)) == 1:
        print("Warning: The model is predicting only one class. It might be overfitting or there's an issue with the data.")

    return model