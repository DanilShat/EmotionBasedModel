from sklearn.base import BaseEstimator, ClusterMixin
from transformers import pipeline, AutoModelForSequenceClassification
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import logging
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA



logging.basicConfig(level=logging.INFO)


class EBTextClusterization(BaseEstimator, ClusterMixin):
    """
    Emotion Based Clusterization.
    This clusterization uses a set of models and a final estimator to make predictions.
    """
    def __init__(self, models, final_estimator):
        self.models = [pipeline(task, model=model_name) for task, model_name in models]
        self.final_estimator = final_estimator
        self.labels = [AutoModelForSequenceClassification.from_pretrained(model_name).config.id2label for _, model_name in models]
        self.labels = [{v: k for k, v in label_dict.items()} for label_dict in self.labels]
        self.vectorizer = TfidfVectorizer()

        
    def fit(self, X):
        """
        Fit the model according to the given training data.
      
        Parameters:
        X : texts or array-like of shape (n_samples, n_features)
            Training vector, where n_samples is the number of samples and n_features is the number of features.
      
        Returns:
        self : object
            Returns self.
        """
        logging.info("Starting fit process...")

        self.vectorizer.fit(X)
        X_transformed = self.transform(X)
        logging.info("Finished transforming X for fit. Starting training final estimator...")
        self.final_estimator.fit(X_transformed)
        logging.info("Finished fit process.")
        return self


    def predict(self, X):
        """
        Predict class labels for samples in X.
      
        Parameters:
        X : texts or array-like of shape (n_samples, n_features)
            The input samples.
      
        Returns:
        y : array-like of shape (n_samples,)
            The predicted values.
        """
        logging.info("Starting predict process...")
        X_transformed = self.transform(X)
        logging.info("Finished transforming X for predict. Starting prediction with final estimator...")
        predictions = self.final_estimator.predict(X_transformed)
        logging.info("Finished predict process.")
        return predictions


    def transform(self, X):
        """
        Transform the input samples.

        Parameters:
        X : array-like of shape (n_samples,)
            The input samples.

        Returns:
        X_transformed : array-like of shape (n_samples, n_features + 2*n_models)
            The transformed input samples. Each sample is transformed into a list of features, where the first n_features are the TF-IDF vectorized features of the sample, and the next 2*n_models features are the labels and scores predicted by the models for that sample.
        """
        logging.info("Starting transform process...")

        # Vectorize X
        X_vectorized = self.vectorizer.transform(X).toarray()

        def transform_sample(sample):
            return [item for sublist in [(self.labels[i][model(sample)[0]['label']], model(sample)[0]['score']) for i, model in enumerate(self.models)] for item in sublist]

        with ThreadPoolExecutor() as executor:
            X_model_transformed = list(tqdm(executor.map(transform_sample, X), total=len(X), desc="Transforming"))

        X_model_transformed = np.array(X_model_transformed)

        X_transformed = np.hstack((X_vectorized, X_model_transformed))

        logging.info("Finished transform process.")

        return X_transformed

  
    def score(self, X):
        """
        Evaluate multiple clustering metrics.
    
        Parameters:
        X : array-like of shape (n_samples, n_features)
            The input samples.
    
        Returns:
        metrics : dict
            A dictionary where the key is the metric name and the value is the metric result.
        """
        y_pred = self.predict(X)
        score = {
            'silhouette_score': silhouette_score(X, y_pred),
            'calinski_harabasz_score': calinski_harabasz_score(X, y_pred),
            'davies_bouldin_score': davies_bouldin_score(X, y_pred)
        }
        return score

          
    def save_model(self, filename):
        """
        Save the model to a file.
    
        Parameters:
        filename : str
            The name of the file where the model will be saved. The filename should end with '.joblib' or '.pkl'.
    
        Returns:
        None
        """
        dump(self, filename)
        logging.info(f'Model saved to {filename}')

    def load_model(filename):
        """
        Load the model from a file.
    
        Parameters:
        filename : str
            The name of the file from where the model will be loaded. The filename should end with '.joblib' or '.pkl'.
    
        Returns:
        model : EBClassifier object
            The loaded model.
        """
        model = load(filename)
        logging.info(f'Model loaded from {filename}')
        return model
