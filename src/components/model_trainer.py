import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join('artifacts', 'model.pkl')

    def initiate_model_trainer(self, train_arr, test_arr):
        print("Starting model training...")
        try:
            X_train, y_train = train_arr[:, :-1], train_arr[:, -1]
            X_test, y_test = test_arr[:, :-1], test_arr[:, -1]

            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)

            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            print(f"Model Accuracy: {accuracy:.4f}")
            print("Classification Report:")
            print(classification_report(y_test, y_pred, target_names=['C', 'M', 'X']))

            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            with open(self.model_path, 'wb') as f:
                pickle.dump(model, f)

            print("Model saved to artifacts/model.pkl")

            return accuracy
        except Exception as e:
            print(f"Error in model trainer: {e}")
            raise e
