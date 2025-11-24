import mlflow
import logging
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',handlers=[logging.FileHandler("training.log"), logging.StreamHandler()])
logging.info("Starting model training...")
logging.info("Loading dataset...")


iris = load_iris()
x_train, x_test, y_train, y_test = train_test_split(
    iris.data, iris.target, test_size=0.2, random_state=42
)
logging.info("Dataset loaded and split into train and test sets.")  

with mlflow.start_run():
    logging.info("Training RandomForestClassifier...")
    model = RandomForestClassifier()
    model.fit(x_train, y_train)
    logging.info("Model training completed.")
    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)
    mlflow.log_metric("accuracy", accuracy)
    logging.info(f"Model accuracy: {accuracy:.4f}")
