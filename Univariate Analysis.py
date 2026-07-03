import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
titanic=sns.load_dataset('titanic')
sns.histplot(titanic['age'].dropna(),bins=30,kde=True)
plt.title('Age Distribution')
plt.show()
