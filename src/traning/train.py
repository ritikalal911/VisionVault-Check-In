from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import KFold
import numpy as np
import pickle
from src.traning.architecture import Architecture

class Train():

    def __init__(self, args):
        self.args = args
        self.data = pickle.load(open(self.args['embeddings'], 'rb'))

    def trainModel(self):
        pass