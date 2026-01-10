import pandas as pd 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

def model_generator(X: pd.DataFrame, Y: pd.Series, \
                    size: float, seed: int, \
                    n: int,) -> tuple:
    """
    Train a KNN model with a fixed number of neighbors.

    Inputs:
        - X (pd.DataFrame): Feature matrix.
        - Y (pd.Series): Target variable.
        - size (float): Test set size as a proportion.
        - seed (int): Random seed for reproducibility.
        - n (int): Number of neighbors for KNN.

    Outputs:
        - model: The trained KNN model.
        - metrics (dict): Evaluation metrics of the model.
    """

    # Train the model
    model = KNeighborsClassifier(n_neighbors=n)
    model.fit(X, Y)

    # Evaluate the model on training data
    y_pred = model.predict(X)

    # Calculate evaluation metrics
    metrics = {
        "accuracy": model.score(X, Y),
        "f1_score": f1_score(Y, y_pred),
        "precision": precision_score(Y, y_pred),
        "recall": recall_score(Y, y_pred)
    }

    return model, metrics