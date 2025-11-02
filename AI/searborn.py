import seaborn as sns
import matplotlib.pyplot as plt

notlar = [60,70,70,70,80,90,85,75,95,100,85,70,56,77,92,74]

sns.histplot(notlar,bins=5,kde=True)
plt.show()

"""veri_setleri = sns.get_dataset_names()
print(veri_setleri)
"""
df = sns.load_dataset("tips")
print(df.head())
#sns.scatterplot(data=df, x="total_bill",y="tip")
#sns.scatterplot(data=df,x="total_bill",y="tip",hue="sex")
sns.barplot(data=df, x="day", y="tip")
plt.show()

corr = df.corr(numeric_only=True)
sns.heatmap(corr,annot=True,cmap="coolwarm")
plt.show()