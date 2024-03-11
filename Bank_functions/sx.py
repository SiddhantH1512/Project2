from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTEN
from imblearn.pipeline import Pipeline as ImPipeline
 

# Create dummy data
X, y = make_classification(n_classes=2, class_sep=2, weights=[0.1, 0.9], n_informative=3, n_redundant=1, flip_y=0, n_features=20, n_clusters_per_class=1, n_samples=1000, random_state=10)

# Define a simple pipeline
pipeline = ImPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTEN(random_state=42))
])

# Test fit_transform
try:
    X_res, y_res = pipeline.fit_transform(X, y)
    print("Pipeline test successful. Shape of X_res:", X_res.shape)
except AttributeError as e:
    print("Pipeline test failed:", e)
