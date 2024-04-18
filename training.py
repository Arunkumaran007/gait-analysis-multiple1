import numpy as np
import pandas as pd
from hmmlearn import hmm
import joblib

# Load dataset
data = pd.read_csv('gaitdataset.csv')
X = data.iloc[:, 1:].values
y = data.iloc[:, 0].values

# Train HMM model
n_components = 6
covariance_type = "full"
n_iter = 100
model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type, n_iter=n_iter)
model.fit(X)

# Save trained model
model_filename = 'hmm_model.joblib'
joblib.dump(model, model_filename)
print(f"Trained model saved as '{model_filename}'.")

# Print model parameters
print("\nModel Parameters:")
print(f"Number of Components: {model.n_components}")
print(f"Covariance Type: {model.covariance_type}")
print(f"Number of Iterations: {model.n_iter}")

# Calculate and print accuracy
accuracyscore = model.score(X)
print(f"\nAccuracy of the Training Data: {accuracyscore}")

# Predict hidden states for training data
hidden_states_train = model.predict(X)
print("\nHidden States for Training Data:")
print(hidden_states_train)

# Predict sample probabilities for training data
sample_probabilities_train = model.predict_proba(X)
print("\nSample Probabilities for Training Data:")
print(sample_probabilities_train)

# Add a new column 'Predicted_Label' to the DataFrame
data['Predicted_Label'] = hidden_states_train

# Save the DataFrame to the same CSV file
data.to_csv('gaitdataset2.csv', index=False)

# Print predictions for a new input
new_input = np.array([[1.68, 70, 2.5, 33, 18, 10, 10.98, 1.204, 0]])
predicted_label = model.predict(new_input)
print("\nPredicted Label for New Input:", predicted_label[0])

predicted_probabilities = model.predict_proba(new_input)
print("\nPredicted Probabilities for New Input:")
print(predicted_probabilities)

# Print model parameters summary
print("\nModel Parameters Summary:")
print("Means:")
print(model.means_)
print("\nCovariances:")
print(model.covars_)
print("\nTransition Probabilities:")
print(model.transmat_)
