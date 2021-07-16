from argparse import Namespace
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 参数
args = Namespace(
    seed=1234,
    data_file="titanic.csv",
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
    max_depth=4,
    min_samples_leaf=5,
    n_estimators=10, # 随机森林中包含的决策树个数
)

# 设置随即种子来保证实验结果的可重复性。
np.random.seed(args.seed)
# 导入csv数据
df = pd.read_csv("./dataset/titanic/train.csv", header=0)
# 前五项
# print(df.head())


# 预处理
def preprocess(df):
    # 删除掉含有空值的行
    df = df.dropna()

    # 删除基于文本的特征 (我们以后的课程将会学习怎么使用它们)
    features_to_drop = ["Name", "Cabin", "Ticket"]
    df = df.drop(features_to_drop, axis=1)

    # pclass, sex, 和 embarked 是类别变量
    # 我们将把字符串转化成浮点数，不再是逻辑回归中的编码变量
    df['Sex'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)
    df["Embarked"] = df['Embarked'].dropna().map({'S': 0, 'C': 1, 'Q': 2}).astype(int)

    return df

# 数据预处理
df = preprocess(df)
# print(df.head())

# 划分数据到训练集和测试集
mask = np.random.rand(len(df)) < args.train_size
train_df = df[mask]
test_df = df[~mask]
# print ("Train size: {0}, test size: {1}".format(len(train_df), len(test_df)))

# 分离 X 和 y
X_train = train_df.drop(["Survived"], axis=1) # 去掉存活信息
y_train = train_df["Survived"]# 拿出存活信息
X_test = test_df.drop(["Survived"], axis=1) # 去掉存活信息
y_test = test_df["Survived"]# 拿出存活信息

# 初始化模型 - 决策树
dtree = DecisionTreeClassifier(criterion="entropy", random_state=args.seed,
                               max_depth=args.max_depth,
                               min_samples_leaf=args.min_samples_leaf)
# 训练
dtree.fit(X_train, y_train)
# 预测
pred_train = dtree.predict(X_train)
pred_test = dtree.predict(X_test)
# 正确率
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
# print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))

# 可解释性 插件无法安装
# dot_data = StringIO()
# export_graphviz(dtree, out_file=dot_data,
#                 feature_names=list(train_df.drop(['Survived'], axis=1)),
#                 class_names = ['died', 'survived'],
#                 rounded = True, filled= True, special_characters=True)
# graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
# Image(graph.create_png(), width=500, height=300)

# 特征重要性
features = list(X_test.columns)
importances = dtree.feature_importances_
indices = np.argsort(importances)[::-1]
num_features = len(importances)

# 画出树中的特征重要性
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(num_features), importances[indices], color="g", align="center")
# plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
# plt.xlim([-1, num_features])
# plt.show()

# # 打印值
# for i in indices:
#     print ("{0} - {1:.3f}".format(features[i], importances[i]))

# 初始化随机森林
forest = RandomForestClassifier(
    n_estimators=args.n_estimators, criterion="entropy",
    max_depth=args.max_depth, min_samples_leaf=args.min_samples_leaf)

# 训练
forest.fit(X_train, y_train)
# 预测
pred_train = forest.predict(X_train)
pred_test = forest.predict(X_test)
# 正确率
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
# print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))

# 计算其他评估指标
precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary")
# print ("precision: {0:.2f}. recall: {1:.2f}, F1: {2:.2f}".format(precision, recall, F1))

# 特征重要性
features = list(X_test.columns)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_], axis=0)
indices = np.argsort(importances)[::-1]
num_features = len(importances)

# 画出树中的特征重要性
# plt.figure()
# plt.title("Feature importances")
# plt.bar(range(num_features), importances[indices], yerr=std[indices],
#         color="g", align="center")
# plt.xticks(range(num_features), [features[i] for i in indices], rotation='45')
# plt.xlim([-1, num_features])
# plt.show()
#
# # 打印
# for i in indices:
#     print ("{0} - {1:.3f}".format(features[i], importances[i]))

# 创建网格的参数
param_grid = {
    'bootstrap': [True],
    'max_depth': [10, 20, 50],
    'max_features': [len(features)],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [4, 8],
    'n_estimators': [5, 10, 50] # of trees
}

# 初始化随机森林
forest = RandomForestClassifier()

# 实例化网格搜索
grid_search = GridSearchCV(estimator=forest, param_grid=param_grid, cv=3,
                           n_jobs=-1, verbose=1)

# 网格搜索拟合数据
grid_search.fit(X_train, y_train)

# 查看最佳参数组合
# print(grid_search.best_params_)

# 使用最佳参数训练
best_forest = grid_search.best_estimator_
best_forest.fit(X_train, y_train)

# 正确率
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))

# 计算其他评价指标
precision, recall, F1, _ = precision_recall_fscore_support(y_test, pred_test, average="binary")
print ("precision: {0:.2f}. recall: {1:.2f}, F1: {2:.2f}".format(precision, recall, F1))



