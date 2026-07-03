import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
titanic=sns.load_dataset('titanic')
print(titanic.isnull().sum())
sns.heatmap(titanic.isnull(),cbar=False,yticklabels=False,cmap='viridis')
plt.title('Missing values Heatmap')
plt.show()
