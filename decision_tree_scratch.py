import numpy as np

class Node:
    def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
        '''
        Constructor for Node in the Decision Tree

        Parameters
        ----------
        1. For Decision Nodes (Internal Nodes):
        - feature_index: The index of the feature used for splitting (e.g., Column 2 "Age").
        - threshold: The value used to split the data (e.g., Age > 30).
        - left: The left child node (Data that meets the condition).
        - right: The right child node (Data that does not meet the condition).
        
        2. For Leaf Nodes (Terminal Nodes):
        - value: The predicted class label (e.g., 1 for "Fraud", 0 for "Legit").
        '''

        # Attributes for Decision Nodes
        self.feature_index = feature_index
        self.threshold = threshold
        self.left = left
        self.right = right

        # Attribute for Leaf Nodes (Only holds a value if it is a leaf)
        self.value = value
    
def calculate_gini(y):
    '''
    Calculate the Gini Impurity for a list of labels.
    Formula: Gini = 1 - sum(probability_of_class_i ^ 2)

    Parameters:
    -----------
    y: array-like, The target labels (e.g., [0, 0, 1, 1, 1])

    Returns:
    --------
    float: The Gini impurity score (0.0 = Pure, 0.5 = Impure/Random for binary)
    '''
    # 1. Get unique class labels (e.g., [0, 1])
    class_labels = np.unique(y)

    # 2. Count total samples
    n_samples = len(y)

    gini_sum = 0

    #3. Apply formula: sum(p^2)
    for cls in class_labels:
        #Calculate probability p for current class
        p_cls = len(y[y == cls]) / n_samples
        gini_sum += p_cls **2

    return 1-gini_sum

def split_data(X, y, feature_index, threshold):
    '''
    Split the dataset into two subsets (left and right) based on a specific feature and threshold.
    
    Parameters:
    -----------
    X: The input features (2D array).
    y: The target labels.
    feature_index: The column index of the feature we are testing (e.g., 0 for "Height").
    threshold: The value to split at (e.g., 170 cm).
    
    Returns:
    --------
    X_left, y_left, X_right, y_right: The split data subsets.
    '''

    # 1. Identify rows that belong to the LEFT (<=threshold)
    left_mask = X[:, feature_index] <= threshold

    # 2. Identify rows that belong to the RIGHT (>threshold)
    right_mask = X[:, feature_index] > threshold

    # 3. Use the masks to split X and y
    # X[left_mask] means "give me only the rows where left_mask is True"
    return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

def information_gain (parent_y, left_y, right_y):
    '''
    Calculate Information Gain
    Logic: Gain = Parent_Gini - (Weighted_Average_Child_Gini)
    '''
    # Calculate weight (proportion) of samples in each child
    weight_l = len(left_y) / len(parent_y)
    weight_r = len(right_y) / len(parent_y)
    
    # Formula: Parent Gini - (Weight_L * Gini_L + Weight_R * Gini_R)
    gain = calculate_gini(parent_y) - (weight_l * calculate_gini(left_y) + weight_r * calculate_gini(right_y))
    return gain
    
def get_best_split(X, y):
    '''
    Find the best feature and threshold to split the data.
    It iterates through EVERY feature and EVERY unique value.
    '''
    best_split = {}
    max_gain = -float("inf") # Start with a very low number
    
    # Loop 1: Iterate over every feature (e.g., 0:Height, 1:Weight)
    num_features = X.shape[1]
    for feature_index in range(num_features):
        
        # Loop 2: Iterate over every unique value in that feature (Potential thresholds)
        # e.g., for Height, try splitting at 150, 160, 180...
        thresholds = np.unique(X[:, feature_index])
        
        for threshold in thresholds:
            # 1. Try splitting!
            X_L, y_L, X_R, y_R = split_data(X, y, feature_index, threshold)
            
            # 2. Skip if split is invalid (one side is empty)
            if len(y_L) == 0 or len(y_R) == 0:
                continue
            
            # 3. Calculate Gain
            gain = information_gain(y, y_L, y_R)
            
            # 4. If this split is the best so far, save it!
            if gain > max_gain:
                best_split = {
                    "feature_index": feature_index,
                    "threshold": threshold,
                    "gain": gain,
                    "X_left": X_L,
                    "y_left": y_L,
                    "X_right": X_R,
                    "y_right": y_R
                }
                max_gain = gain
                
    return best_split


class DecisionTree:
    def __init__(self, min_samples_split = 2, max_depth = 100):
        '''
        Decision Tree Classifier.
        - min_samples_split: If a node has fewer samples than this, stop splitting. (Prevent Overfitting!)
        - max_depth: The maximum depth of the tree. (Prevent Overfitting!)
        '''
        self.root = None
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth

    def build_tree(self, X, y, depth = 0):
        '''
        Recursive function to build the tree.
        '''
        num_samples, num_features = X.shape
        num_labels = len(np.unique(y))

        # 1. Stopping Criteria (When to stop?)
        # - Too deep (depth > max_depth)
        # - Too less people (num_samples < min_samples_split)
        # - Already pure (num_labels == 1)
        if (depth >= self.max_depth or num_labels == 1 or num_samples < self.min_samples_split):
            leaf_value = self._calculate_leaf_value(y)
            return Node(value=leaf_value) # Return a Leaf Node (Answer)
        
        # 2. Find the best split
        best_split = get_best_split(X, y)

        # 3. If gain is positive (useful split), keep building!
        if best_split.get("gain", 0) > 0:
            # Recursion: Build Left Subtree
            left_subtree = self.build_tree(best_split["X_left"], best_split["y_left"], depth + 1)
            
            # Recursion: Build Right Subtree
            right_subtree = self.build_tree(best_split["X_right"], best_split["y_right"], depth + 1)
            
            # Return a Decision Node
            return Node(feature_index=best_split["feature_index"], 
                        threshold=best_split["threshold"], 
                        left=left_subtree, 
                        right=right_subtree)
        # 4. If gain is 0 (can't improve), tuen into leaf
        leaf_value = self._calculate_leaf_value(y)
        return Node(value=leaf_value)
    
    def _calculate_leaf_value(self, y):
        '''
        Helper: Find the most common label in the list y.
        e.g., [1, 1, 1, 0] -> Returns 1
        '''
        Y = list(y)
        return max(Y, key=Y.count)
    
    def fit(self, X, y):
        '''
        Start training the tree.
        '''
        # Note: We added the "X_left" and "X_right" to best_split dictionary in previous step conceptually,
        # But we need to make sure get_best_split actually returns the data subsets.
        # Let's assume get_best_split is updated or we handle it here.
        # actually, let's keep it simple. We need to update get_best_split slightly to return X_L, y_L...
        # Wait, to make it easier for you, I will ask you to update get_best_split slightly below.
        
        self.root = self.build_tree(X, y)

    def print_tree(self, tree=None, indent=" "):
        '''
        Visualizer: Print the tree structure.
        '''
        if not tree:
            tree = self.root
        
        if tree.value is not None:
            print(tree.value)
        
        else:
            print("X_" + str(tree.feature_index), "<=", tree.threshold, "?")
            print("%sleft:" % (indent), end="")
            self.print_tree(tree.left, indent + indent)
            print("%sright:" % (indent), end="")
            self.print_tree(tree.right, indent + indent)

# --- Test Zone ---
if __name__ == "__main__":
    # Fake Dataset: [Height(cm), Weight(Kg)]
    X_fake = np.array([
        [150, 45],
        [160, 50], 
        [180, 80], 
        [190, 90]
    ])
    y_fake = np.array([0, 0, 1, 1]) 
    
    print("\n--- Building the Full Tree ---")
    classifier = DecisionTree(min_samples_split=2, max_depth=3)
    classifier.fit(X_fake, y_fake)
    
    print("Tree Structure:")
    classifier.print_tree()