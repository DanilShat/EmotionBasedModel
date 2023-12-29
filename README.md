# EmotionBasedModel

EmotionBasedModel is a Python package designed for machine learning tasks on smaller datasets where fine-tuning might not be the optimal choice due to computational constraints. The package includes models for classification, regression, and clustering tasks.

The core concept of EmotionBasedModel is inspired by human emotions and their impact on decision-making. Just as humans react to stimuli based on the emotions they evoke, EmotionBasedModel leverages the outputs of various pre-trained models as "emotions" to guide its predictions.

For instance, consider how a person reacts to a red traffic light. The color red often evokes a sense of danger or caution, prompting the person to stop. Similarly, EmotionBasedModel uses the results from different models (akin to "seeing red") to train a new decision model. These "emotions" extracted from the pre-trained models serve as a powerful and efficient way to remember and learn from the data.

This approach makes EmotionBasedModel particularly effective and computationally efficient for smaller datasets, providing a unique and innovative solution for machine learning tasks.

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install EmotionBasedModel.

```bash
pip install EmotionBasedModel
```

## Usage

```python
from EmotionBasedModel import EBTextClassifier, EBTextRegressor, EBTextClusterization

# your code here
```

## Models

### EBTextClassifier

This is EmotionBased model for Text Classification.

#### Methods

##### fit(self, X, y)
Fit the model according to the given training data.
      
Parameters:
X : texts or array-like of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and n_features is the number of features.
y : array-like of shape (n_samples,)
    Target vector relative to X.
      
Returns:
  self : object
  Returns self.

##### predict(self, X)

Predict class labels for samples in X.
      
Parameters:
  X : texts or array-like of shape (n_samples, n_features)
      The input samples.
      
Returns:
  y : array-like of shape (n_samples,)
      The predicted class labels.

##### transform(self, X)

Transform the input samples.

Parameters:
  X : array-like of shape (n_samples,)
      The input samples.

Returns:
  X_transformed : array-like of shape (n_samples, n_features + 2*n_models)
  The transformed input samples. Each sample is transformed into a list of features, where the first n_features are the TF-IDF vectorized features of the sample, and the next 2*n_models features are the labels and scores predicted by the models for that sample.

##### score(self, X, y_true)

Calculate accuracy score.
          
Parameters:
  X : array-like of shape (n_samples, n_features)
      The input samples.
  y_true : array-like of shape (n_samples,)
      The true labels for X.
  
Returns:
  score : float
  The accuracy score.

##### save_model(self, filename)

Save the model to a file.
    
Parameters:
  filename : str
      The name of the file where the model will be saved. The filename should end with '.joblib' or '.pkl'.
    
Returns:
  None

##### load_model(filename)

Load the model from a file.
    
Parameters:
  filename : str
  The name of the file from where the model will be loaded. The filename should end with '.joblib' or '.pkl'.
    
Returns:
  model : EBClassifier object
  The loaded model.

### EBTextRegressor

This is EmotionBased model for Text Regression tasks.

#### Methods

##### fit(self, X, y)
Fit the model according to the given training data.
      
Parameters:
X : texts or array-like of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and n_features is the number of features.
y : array-like of shape (n_samples,)
    Target vector relative to X.
      
Returns:
  self : object
  Returns self.

##### predict(self, X)

Predict values for samples in X.
      
Parameters:
  X : texts or array-like of shape (n_samples, n_features)
      The input samples.
      
Returns:
  y : array-like of shape (n_samples,)
      The predicted values.

##### transform(self, X)

Transform the input samples.

Parameters:
  X : array-like of shape (n_samples,)
      The input samples.

Returns:
  X_transformed : array-like of shape (n_samples, n_features + 2*n_models)
  The transformed input samples. Each sample is transformed into a list of features, where the first n_features are the TF-IDF vectorized features of the sample, and the next 2*n_models features are the labels and scores predicted by the models for that sample.

##### score(self, X, y_true)

Evaluate multiple metrics.
    
Parameters:
  X : array-like of shape (n_samples, n_features)
      The input samples.
  y_true : array-like of shape (n_samples,)
      The true labels for X.
    
Returns:
  score : dict
  A dictionary where the key is the metric name and the value is the metric result.

##### save_model(self, filename)

Save the model to a file.
    
Parameters:
  filename : str
      The name of the file where the model will be saved. The filename should end with '.joblib' or '.pkl'.
    
Returns:
  None

##### load_model(filename)

Load the model from a file.
    
Parameters:
  filename : str
  The name of the file from where the model will be loaded. The filename should end with '.joblib' or '.pkl'.
    
Returns:
  model : EBClassifier object
  The loaded model.

##### visualize(self, X, y_true)

Visualize the results of the regression.
    
Parameters:
  X : array-like of shape (n_samples, n_features)
      The input samples.
  y_true : array-like of shape (n_samples,)
      The true values for X.

### EBTextClusterization

This is EmotionBased model for Text Clusterization tasks.

#### Methods

##### fit(self, X, y)
Fit the model according to the given training data.
      
Parameters:
X : texts or array-like of shape (n_samples, n_features)
    Training vector, where n_samples is the number of samples and n_features is the number of features.
      
Returns:
  self : object
  Returns self.

##### predict(self, X)

Predict values for samples in X.
      
Parameters:
  X : texts or array-like of shape (n_samples, n_features)
      The input samples.
      
Returns:
  y : array-like of shape (n_samples,)
      The predicted values.

##### transform(self, X)

Transform the input samples.

Parameters:
  X : array-like of shape (n_samples,)
      The input samples.

Returns:
  X_transformed : array-like of shape (n_samples, n_features + 2*n_models)
  The transformed input samples. Each sample is transformed into a list of features, where the first n_features are the TF-IDF vectorized features of the sample, and the next 2*n_models features are the labels and scores predicted by the models for that sample.

##### score(self, X, y_true)

Evaluate multiple clustering metrics.
    
Parameters:
  X : array-like of shape (n_samples, n_features)
      The input samples.
    
Returns:
  metrics : dict
  A dictionary where the key is the metric name and the value is the metric result.

##### save_model(self, filename)

Save the model to a file.
    
Parameters:
  filename : str
      The name of the file where the model will be saved. The filename should end with '.joblib' or '.pkl'.
    
Returns:
  None

##### load_model(filename)

Load the model from a file.
    
Parameters:
  filename : str
  The name of the file from where the model will be loaded. The filename should end with '.joblib' or '.pkl'.
    
Returns:
  model : EBClassifier object
  The loaded model.
