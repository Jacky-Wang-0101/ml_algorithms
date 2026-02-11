import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Import the DecisionTree class you wrote in the other file
# Ensure decision_tree_scratch.py is in the same folder!
from decision_tree_scratch import DecisionTree

def predict_single_input(node, x):
    '''
    Helper function to traverse the tree and predict the label for a single input 'x'.
    Since we haven't implemented a .predict() method in our class yet, 
    we use this helper function for testing.
    '''
    # Base case: If the node has a value, it is a Leaf Node. Return the prediction.
    if node.value is not None:
        return node.value
    
    # Recursive step: Decide whether to go left or right based on the threshold
    feature_val = x[node.feature_index]

    if feature_val <= node.threshold:
        return predict_single_input (node.left, x)
    else:
        return predict_single_input (node.right, x)
    
# --- Main Execution Block ---
if __name__ == "__main__":

    # 1. Load the Iris Dataset (Real World Data)
    # X contains 4 features: Sepal Length, Sepal Width, Petal Length, Petal Width
    # y contains 3 classes: 0 (Setosa), 1 (Versicolor), 2 (Virginica)
    print("--- Loading Iris Dataset ---")
    iris = load_iris()
    X = iris.data
    y = iris.target

    print(f"Dataset Shape: {X.shape} (150 samples, 4 features)")
    print(f"Target Classes: {np.unique(y)}")

    # 2. Split data into Training set (80%) and Test Set (20%)
    # random_state=42 ensures we get the same split every time we run it
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"Training Samples: {len(X_train)}")
    print(f"Testing Samples: {len(X_test)}")

    # 3. Initialize and Train the Model
    # max_depth=3 means the tree can only grow 3 levels deep (prevents overfitting)
    print("\n--- Building the Decision Tree ---")
    model = DecisionTree(min_samples_split=2, max_depth=3)
    model.fit(X_train, y_train)

    # 4. Visualize the Tree Structure
    print("\n--- Tree Structure (Visualized) ---")
    # Note: Feature indices (0, 1, 2, 3) correspond to columns in X
    model.print_tree()

    # Apply our helper function to every row in the test set
    y_pred = [predict_single_input(model.root, row) for row in X_test]

    # Calculate Accuracy
    acc = accuracy_score(y_test, y_pred)

    # Show first 10 predictions vs real answers
    print(f"True Labels:      {y_test[:10]} ...")
    print(f"Predicted Labels: {y_pred[:10]} ...")
    print(f"Final Accuracy: {acc * 100:.2f}%")