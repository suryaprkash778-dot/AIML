import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cufflinks as cf
train = pd.read_csv("titanic_train.csv")

print(train.head())

#sns.heatmap(train.isnull(),yticklabels=False,cbar=False)

sns.set_style('whitegrid')
#sns.countplot(x='Survived',hue='Pclass',data=train,)
#sns.histplot(train['Age'].dropna(),kde=False,bins=30)

#sns.countplot(x='SibSp',data=train,)
#train['Fare'].hist(bins=40,figsize=(10,4))
cf.go_offline()
train['Fare'].iplot(kind='hist',bins=30)
train.info()
plt.show()
