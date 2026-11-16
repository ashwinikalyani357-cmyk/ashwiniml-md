import os
import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import load_breast_cancer
from predict import make_prediction
MODEL_PATH = "best_model.joblib"

#FIXTURE 1:Load the trainedmodel
@pytest.fixture
def model():
  assert os.path.exists(MODEL_PATH),"Model file not found!"
  return joblib.load(MODEL_PATH)
