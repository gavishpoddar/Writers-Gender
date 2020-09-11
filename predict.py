import numpy as np 
from features import extract_features
from helper import sigmoid

weights = np.load("data/model_weights.npy")
biases = np.load("data/model_biases.npy")

def predict(text):
  score = sigmoid(np.dot(extract_features(text), weights) + biases)
  rv = {
    "score": score,
  }
  if score > 0.5:
    rv['gender'] = "Male"
  else:
    rv['gender'] = "Female"

  return rv
