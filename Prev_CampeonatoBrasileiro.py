
import pandas as pd
import numpy as np
from sklearn.linear_model import  LogisticRegression  # Modeles 
from sklearn.tree import DecicisioClassfier #Metricas
from sklearn.naives_bayes import GaussianNb #Metricas
from sklearn.svm import SVC 
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKbest
from sklearn.preprocessing import StandardScaler, MixMaxScaler
from Ipython.display import display
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import scale

%matplotlib inline