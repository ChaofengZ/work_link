from argparse import Namespace
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 调包
from sklearn.linear_model.stochastic_gradient import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 参数
args = Namespace(
    seed=1234,
    data_file="sample_data.csv",
    num_samples=100,
    train_size=0.75,
    test_size=0.25,
    num_epochs=100,
)

# 设置随机种子来保证实验结果的可重复性。
np.random.seed(args.seed)

# 生成数据
def generate_data(num_samples):
    X = np.array(range(num_samples))
    y = 3.65*X + 10
    return X, y

# 生成随机数据
X, y = generate_data(args.num_samples)
data = np.vstack([X, y]).T
df = pd.DataFrame(data, columns=['X', 'y'])
df.head()

# 画散点图
# plt.title("Generated data")
# plt.scatter(x=df["X"], y=df["y"])
# plt.show()


# 划分数据到训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    df["X"].values.reshape(-1, 1), df["y"], test_size=args.test_size,
    random_state=args.seed)
# print ("X_train:", X_train.shape)
# print ("y_train:", y_train.shape)
# print ("X_test:", X_test.shape)
# print ("y_test:", y_test.shape)

# 标准化训练集数据 (mean=0, std=1)
X_scaler = StandardScaler().fit(X_train)
y_scaler = StandardScaler().fit(y_train.values.reshape(-1,1))
# print(X_scaler )
# print(y_scaler )

# 注意此处不严谨，严格来说应该先将整个数据集标准化后在对其进行划分训练集和数据集

# 在训练集和测试集上进行标准化操作
standardized_X_train = X_scaler.transform(X_train)
standardized_y_train = y_scaler.transform(y_train.values.reshape(-1,1)).ravel()
standardized_X_test = X_scaler.transform(X_test)
standardized_y_test = y_scaler.transform(y_test.values.reshape(-1,1)).ravel()

# 检查
# print ("mean:", np.mean(standardized_X_train, axis=0),
#        np.mean(standardized_y_train, axis=0)) # mean 应该是 ~0
# print ("std:", np.std(standardized_X_train, axis=0),
#        np.std(standardized_y_train, axis=0))   # std 应该是 1

# 初始化模型
lm = SGDRegressor(loss="squared_loss", penalty="none", max_iter=args.num_epochs)
# 训练
lm.fit(X=standardized_X_train, y=standardized_y_train)

# 预测 (还未标准化)
pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_


print(X_train,pred_train)

# 训练和测试集上的均方误差 MSE
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
# print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(train_mse, test_mse))

# # 图例大小
# plt.figure(figsize=(15,5))
#
# # 画出训练数据
# plt.subplot(1, 2, 1)
# plt.title("Train")
# plt.scatter(X_train, y_train, label="y_train")
# plt.plot(X_train, pred_train, color="red", linewidth=1, linestyle="-", label="lm")
# plt.legend(loc='lower right')
#
# # 画出测试数据
# plt.subplot(1, 2, 2)
# plt.title("Test")
# plt.scatter(X_test, y_test, label="y_test")
# plt.plot(X_test, pred_test, color="red", linewidth=1, linestyle="-", label="lm")
# plt.legend(loc='lower right')
#
# # 显示图例
# plt.show()

# 传入我们自己的输入值
X_infer = np.array((0, 1, 2), dtype=np.float32)
standardized_X_infer = X_scaler.transform(X_infer.reshape(-1, 1))
pred_infer = (lm.predict(standardized_X_infer) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
# print (pred_infer)
df.head(3)

# 未标准化系数
coef = lm.coef_ * (y_scaler.scale_/X_scaler.scale_)
intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - np.sum(coef*X_scaler.mean_)
# print (coef) # ~3.65
# print (intercept) # ~10

# 初始化带有L2正则化的模型
lm = SGDRegressor(loss="squared_loss", penalty='l2', alpha=1e-2,
                  max_iter=args.num_epochs)

# 训练
lm.fit(X=standardized_X_train, y=standardized_y_train)
# print(lm.fit(X=standardized_X_train, y=standardized_y_train))

# 预测 (还未标准化)
pred_train = (lm.predict(standardized_X_train) * np.sqrt(y_scaler.var_)) + y_scaler.mean_
pred_test = (lm.predict(standardized_X_test) * np.sqrt(y_scaler.var_)) + y_scaler.mean_

# 训练集和测试集的MSE
train_mse = np.mean((y_train - pred_train) ** 2)
test_mse = np.mean((y_test - pred_test) ** 2)
# print ("train_MSE: {0:.2f}, test_MSE: {1:.2f}".format(
#     train_mse, test_mse))

# 未标准化系数
coef = lm.coef_ * (y_scaler.scale_/X_scaler.scale_)
intercept = lm.intercept_ * y_scaler.scale_ + y_scaler.mean_ - (coef*X_scaler.mean_)
# print (coef) # ~3.65
# print (intercept) # ~10


# 创建类别特征
cat_data = pd.DataFrame(['a', 'b', 'c', 'a'], columns=['favorite_letter'])
cat_data.head()
# print(cat_data.head())

dummy_cat_data = pd.get_dummies(cat_data) #独热编码 one-hot encoding，与dummy变量不同要注意。
# print(dummy_cat_data.head())

# todo :尝试使用上述方法训练类别变量
