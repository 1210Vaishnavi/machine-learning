import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
titanic=sns.load_dataset('titanic')
sns.barplot(x='sex',y='survived',data=titanic)
plt.title('Survival Rate by Sex')
plt.show()
