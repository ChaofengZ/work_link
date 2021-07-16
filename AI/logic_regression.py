from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import urllib

# 调包
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

# 参数
args = Namespace(
    seed=1234,
    data_file="titanic.csv",
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
)

# 设置随即种子来保证实验结果的可重复性。
np.random.seed(args.seed)

# 导入csv数据
df = pd.read_csv("./dataset/titanic/train.csv", header=0)
# 前五项
df.head()


# 预处理
def preprocess(df):
    # 删除掉含有空值的行
    df = df.dropna()

    # 删除基于文本的特征 (我们以后的课程将会学习怎么使用它们)
    features_to_drop = ["Name", "Cabin", "Ticket"]
    df = df.drop(features_to_drop, axis=1)

    # pclass, sex, 和 embarked 是类别变量
    categorical_features = ["Pclass", "Embarked", "Sex"]
    df = pd.get_dummies(df, columns=categorical_features)

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
X_train = train_df.drop(["Survived"], axis=1)
y_train = train_df["Survived"]
X_test = test_df.drop(["Survived"], axis=1)
y_test = test_df["Survived"]

# 标准化训练数据 (mean=0, std=1)
X_scaler = StandardScaler().fit(X_train)

# 标准化训练和测试数据  (不要标准化标签分类y)
standardized_X_train = X_scaler.transform(X_train)
standardized_X_test = X_scaler.transform(X_test)

# 检查
# print ("mean:", np.mean(standardized_X_train, axis=0)) # mean 应该为 ~0
# print ("std:", np.std(standardized_X_train, axis=0))   # std 应该为 1

# 初始化模型
log_reg = SGDClassifier(loss="log", penalty="none", max_iter=args.num_epochs,
                        random_state=args.seed)
# 训练
log_reg.fit(X=standardized_X_train, y=y_train)

# 概率
pred_test = log_reg.predict_proba(standardized_X_test)
# print (pred_test[:5])

# 预测 (未标准化)
pred_train = log_reg.predict(standardized_X_train)
pred_test = log_reg.predict(standardized_X_test)
# print (pred_test)

# 正确率
train_acc = accuracy_score(y_train, pred_train)
test_acc = accuracy_score(y_test, pred_test)
# print ("train acc: {0:.2f}, test acc: {1:.2f}".format(train_acc, test_acc))

import itertools
from sklearn.metrics import classification_report, confusion_matrix

# 绘制混淆矩阵
def plot_confusion_matrix(cm, classes):
    cmap=plt.cm.Blues
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title("Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    plt.grid(False)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    # plt.show()

# 混淆矩阵
cm = confusion_matrix(y_test, pred_test)
plot_confusion_matrix(cm=cm, classes=["Died", "Survived"])
# print (classification_report(y_test, pred_test))

# 未标准化系数
coef = log_reg.coef_ / X_scaler.scale_
intercept = log_reg.intercept_ - np.sum((coef * X_scaler.mean_))
# print (coef)
# print (intercept)

indices = np.argsort(coef)
features = list(X_train.columns)
# print ("Features correlated with death:", [features[i] for i in indices[0][:3]])
# print ("Features correlated with survival:", [features[i] for i in indices[0][-3:]])

#  K折交叉验证
log_reg = SGDClassifier(loss="log", penalty="none", max_iter=args.num_epochs)
scores = cross_val_score(log_reg, standardized_X_train, y_train, cv=10, scoring="accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())

# Todo: 看章节对应笔记
