import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
titanic=sns.load_dataset('titanic')
print(titanic.isnull().sum())
sns.heatmap(titanic.isnull(),cbar=False,yticklabels=False,cmap='viridis')
plt.title('Missing values Heatmap')
plt.show()
survived         0
pclass           0
sex              0
age            177
sibsp            0
parch            0
fare             0
embarked         2
class            0
who              0
adult_male       0
deck           688
embark_town      2
alive            0
alone            0
dtype: int64
