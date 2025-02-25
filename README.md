# Software Performance Prediction Framework

## Overview of the Implementation

A Software Performance Prediction Framework that leverages Classification and Regression Trees (CART) and Performance Influence Models (PIM) to analyze and predict software performance across different configurations.

## 1. CART Implementation

### 1.1 Fundamental Concepts

The CART implementation is built on recursive binary partitioning, creating a decision tree where:
- Each internal node represents a decision based on a feature's state (enabled/disabled)
- Each leaf node contains a predicted performance value
- The tree is constructed by minimizing squared error at each split

### 1.2 Key Functions

#### Statistical Utilities
- `calc_mean()`: Calculates the average performance of a set of configurations
- `calc_sqerr()`: Computes the sum of squared errors from the mean

#### Tree Construction
- `leaf_node()`: Creates terminal nodes with performance predictions
- `best_split()`: Identifies the optimal feature for splitting configurations by:
  - Evaluating each feature's potential to minimize error
  - Partitioning data based on feature state (1=enabled, 0=disabled)
  - Selecting the feature that produces minimum aggregate error

#### Recursive Algorithm
The `build_cart()` function implements the core recursive algorithm:
1. Calculates the mean performance of the current configuration set
2. Terminates if error is below threshold (10) or no valid split exists
3. Finds the best feature to split on
4. Recursively builds left subtree (feature enabled) and right subtree (feature disabled)

### 1.3 Error Handling

The implementation handles several edge cases:
- Empty configuration sets
- Configurations with minimal variance (error < 10)
- Cases where no valid split exists

## 2. Performance Influence Model (PIM)

### 2.1 Mathematical Foundation

The PIM represents performance as a sum of:
- Base performance (empty configuration)
- Individual feature influences
- Higher-order interaction effects (pairs and triplets)

### 2.2 Influence Calculation

The implementation builds the model incrementally:
1. `base_perf()`: Extracts baseline performance
2. `single_infl()`: Calculates individual feature influences
3. `double_infl()`: Computes pairwise interaction effects
4. `triple_infl()`: Determines three-way interaction effects

### 2.3 Data Processing

The model ensures accurate interaction calculations by:
- Isolating configurations with specific feature combinations
- Comparing actual performance against expected performance
- Recording only significant interactions (non-zero effects)

## 3. Performance Prediction and Optimization

### 3.1 CART-Based Prediction

The `get_performance()` function navigates the CART based on configuration features:
- Starting at the root
- Following left branch when feature is enabled (in configuration set)
- Following right branch when feature is disabled
- Returning the mean performance of the reached leaf node

### 3.2 Error Rate Calculation

The `get_error_rate()` function:
- Processes each configuration in the dataset
- Predicts performance using the CART
- Computes absolute difference between predicted and actual values
- Returns the average error across all configurations

### 3.3 Configuration Optimization

The `get_optimal_configuration()` function:
1. Identifies all possible feature combinations
2. Filters valid configurations based on feature model constraints
3. Ensures partial configuration requirements are met
4. Calculates cost for each valid configuration using the PIM
5. Returns the configuration with minimum cost

### 3.4 Feature Model Constraints

The implementation supports three types of feature relationships:
- `And`: All mandatory child features must be selected
- `Or`: At least one child feature must be selected
- `Xor`: Exactly one child feature must be selected

## 4. Implementation Details

### 4.1 Data Structures

- CART: Nested dictionary representing the binary decision tree
- PIM: Dictionary mapping feature combinations to performance influences
- Configurations: Sets of enabled features

### 4.2 Performance Considerations

- The implementation uses set operations for efficient configuration comparison
- Features are lexicographically sorted to ensure consistent interaction keys
- Early termination criteria prevent over-fitting in the CART

### 4.3 Extension Points

The framework could be extended by:
- Supporting additional feature model constraints
- Implementing pruning techniques for CART optimization
- Adding higher-order interaction calculations to the PIM

## 5. Usage Example

```python
# 1. Create a CART from performance data
cart = get_cart("performance_data.csv")

# 2. Build a Performance Influence Model
pim = get_pim("performance_data.csv")

# 3. Predict performance of a specific configuration
config = {"FeatureA", "FeatureC"}
performance = get_performance(cart, config)

# 4. Calculate error rate of the CART
error = get_error_rate(cart, "validation_data.csv")

# 5. Find optimal configuration given constraints
feature_model = {...}  # Feature model definition
partial_config = {"MandatoryFeature"}
optimal_config, cost = get_optimal_configuration(pim, feature_model, partial_config)
```
