import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 导入csv数据
df = pd.read_csv("train.csv", header=0)
# 前五项
df.head()

# 描述性统计
df.describe()

# 显示Age数据直方图
# plt.hist(df['Age'],bins = 10, range=[0,100])
# plt.title('Age distribution')
# plt.xlabel('Age')
# plt.ylabel("频数")
# plt.show()

# 唯一值
df["Pclass"].unique()

# 根据特征选择数据
df["Name"].head()

# 筛选
df[df["Sex"]=="female"].head() # 只有女性数据出现

# 排序
df.sort_values("Age", ascending=False).head()

# Grouping（数据聚合与分组运算）
sex_group = df.groupby("Survived")
sex_group.mean()

# iloc根据位置的索引来访问
df.iloc[0, :] # iloc在索引中的特定位置获取行（或列）（因此它只需要整数）

# 获取指定位置的数据
df.iloc[0, 1]

# 具有至少一个NaN值的行
df[pd.isnull(df).any(axis=1)].head()

# 删除具有Nan值的行
df = df.dropna() # 删除具有NaN值的行
df = df.reset_index() # 重置行索引
df.head()

# 删除多行
df = df.drop(["Name", "Cabin", "Ticket"], axis=1) # we won't use text features for our initial basic models
df.head()

# 映射特征值
df['Sex'] = df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)
df["Embarked"] = df['Embarked'].dropna().map( {'S':0, 'C':1, 'Q':2} ).astype(int)
df.head()

# lambda表达式创建新特征
def get_family_size(sibsp, parch):
    family_size = sibsp + parch
    return family_size

df["family_size"] = df[["SibSp", "Parch"]].apply(lambda x: get_family_size(x["SibSp"], x["Parch"]), axis=1)
df.head()

# 重新组织标题
df = df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'family_size', 'Fare', 'Embarked', 'Survived']]
df.head()

# 保存数据帧（dataframe）到 CSV
df.to_csv("processed_titanic.csv", index=False)

print(df.head())
