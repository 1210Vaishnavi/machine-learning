import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
titanic=sns.load_dataset('titanic')
print("Dataset shape:",titanic.shape)
print(titanic.info())
print(titanic.describe())
print(titanic.describe(include=['object','category']))
