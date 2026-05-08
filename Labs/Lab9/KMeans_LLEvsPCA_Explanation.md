# KMeans 与 LLEvsPCA Notebook 说明

这份文档说明两份 notebook 的核心目标、整体流程，以及 TODO 中补全代码的具体作用。

涉及文件：

- `Labs/Lab10/KMeans.ipynb`
- `Labs/Lab11/LLEvsPCA.ipynb`

## 1. Lab10: KMeans.ipynb

### 核心任务

这份 notebook 用 Iris 数据集演示 K-Means 聚类。

Iris 数据集中每条样本有花萼、花瓣等特征，还有真实类别 `target`。本实验只使用两个特征：

- `petal length (cm)`
- `petal width (cm)`

目标是用 K-Means 把样本自动分成 3 类，然后把聚类结果和真实花种标签做对照，计算一个自定义 accuracy。

K-Means 是无监督算法，所以它不会直接知道真实标签。它输出的是 cluster 编号，例如 `0, 1, 2`。这些编号本身没有语义，因此必须先把 cluster 编号映射到真实类别，再计算准确率。

### 整体流程

1. 加载 Iris 数据集，并构造 `df`。
2. 查看数据结构和类别分布。
3. 用 pairplot、jointplot 观察不同类别在特征空间中的分布。
4. 选取花瓣长度和花瓣宽度作为聚类输入。
5. 使用 K-Means 聚成 3 类。
6. 将每个样本的聚类结果保存到 `df['cluster']`。
7. 提取聚类中心 `centroids`，用于后续可视化。
8. 画散点图，展示 cluster 和中心点。
9. 计算混淆矩阵。
10. 将 cluster 编号映射到真实类别，计算 accuracy。

### 补全代码说明

#### 1. 选择聚类特征

```python
X = df[['petal length (cm)', 'petal width (cm)']]
```

这行代码从完整的 Iris 表格中取出两个用于聚类的特征。

K-Means 只会看到这两个维度，而不会使用真实标签 `target`。这符合无监督学习的设定。

#### 2. 创建 K-Means 模型

```python
kmeans = cluster.KMeans(n_clusters=3, random_state=42, n_init=10)
```

这行代码创建 K-Means 模型。

- `n_clusters=3`：Iris 数据集有 3 个真实类别，所以聚成 3 类。
- `random_state=42`：固定随机种子，让结果可复现。
- `n_init=10`：用 10 次不同初始化运行 K-Means，取效果最好的结果，减少随机初始化带来的不稳定。

#### 3. 拟合模型并保存 cluster

```python
df['cluster'] = kmeans.fit_predict(X)
```

这行代码做了两件事：

- `fit`：根据 `X` 学习 3 个聚类中心。
- `predict`：给每个样本分配一个 cluster 编号。

最终得到的 cluster 编号被保存到 `df['cluster']`，后面的散点图、混淆矩阵和 accuracy 都会用到它。

#### 4. 获取聚类中心

```python
centroids = kmeans.cluster_centers_
```

这行代码取出 K-Means 学到的 3 个中心点坐标。

因为聚类只使用了两个特征，所以每个中心点也是二维的：

- 第一维对应 `petal length (cm)`
- 第二维对应 `petal width (cm)`

后续图中用红色三角形标出这些中心点。

#### 5. 构造 cluster 到真实类别的映射

```python
cluster_to_label = {}
for cluster_id in sorted(df['cluster'].unique()):
    true_labels = df.loc[df['cluster'] == cluster_id, 'target']
    cluster_to_label[cluster_id] = mode(true_labels, keepdims=False).mode
```

这段代码解决 K-Means 标签编号无语义的问题。

例如 K-Means 的 `cluster=1` 不一定等于真实类别 `target=1`。所以需要看每个 cluster 里最多的真实类别是什么。

流程是：

1. 遍历所有 cluster 编号。
2. 找出属于当前 cluster 的样本。
3. 取这些样本的真实标签 `target`。
4. 用 `mode` 找出出现次数最多的真实标签。
5. 把当前 cluster 映射到这个真实标签。

举例来说，如果 `cluster=0` 中大多数样本真实标签是 `2`，那么就把 `0 -> 2`。

#### 6. 生成预测标签并计算准确率

```python
df['predicted_target'] = df['cluster'].map(cluster_to_label)
accuracy = accuracy_score(df['target'], df['predicted_target'])
print(f"Accuracy: {accuracy:.2f}")
```

这段代码先把无语义的 cluster 编号转换成真实类别编号，再和 `df['target']` 对比。

- `df['predicted_target']`：映射后的预测类别。
- `accuracy_score`：计算预测类别和真实类别的匹配比例。

当前 notebook 的结果为：

```text
Accuracy: 0.96
```

这说明仅用花瓣长度和花瓣宽度做 K-Means 聚类，已经能很好地区分 Iris 的三个类别。

## 2. Lab11: LLEvsPCA.ipynb

### 核心任务

这份 notebook 比较两种降维方法在人脸识别任务中的表现：

- PCA: Principal Component Analysis，主成分分析。
- LLE: Locally Linear Embedding，局部线性嵌入。

使用的数据集是 Olivetti Faces：

- 共 400 张人脸图像。
- 每张图像大小为 `64 x 64`。
- 展平后每张图像有 `4096` 个特征。
- 共 40 个类别，每个人 10 张图。

实验目标是：

1. 先在原始 4096 维数据上训练 k-NN，得到 baseline。
2. 用 PCA 降维后训练 k-NN。
3. 用 LLE 降维后训练 k-NN。
4. 比较三者的 accuracy 和运行时间。

### 整体流程

1. 加载 Olivetti Faces 数据集。
2. 查看数据形状：`X` 是图像特征，`y` 是人脸类别。
3. 可视化部分样本图像。
4. 画类别直方图，确认类别分布。
5. 使用 `train_test_split` 划分训练集和测试集。
6. 先在原始数据上训练 k-NN，作为 baseline。
7. 搜索不同 PCA 组件数，找到 k-NN accuracy 最好的组件数。
8. 搜索不同 LLE 邻居数和组件数，找到 k-NN accuracy 最好的参数组合。
9. 使用最优参数分别测量 PCA 和 LLE 的降维耗时。
10. 在原始数据、PCA 降维数据、LLE 降维数据上分别测量 k-NN 时间和 accuracy。
11. 用柱状图比较降维时间、k-NN 时间和 accuracy。

### 补全代码说明

#### 1. PCA 参数搜索

```python
best_pca_acc = 0
best_pca_components = 0

for n_components in range(10, 101, 10):
    pca = PCA(n_components=n_components, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    y_pred_pca = knn.predict(X_test_pca)
    acc = accuracy_score(y_test, y_pred_pca)

    if acc > best_pca_acc:
        best_pca_acc = acc
        best_pca_components = n_components
```

这段代码用于寻找 PCA 的最佳降维维度。

核心逻辑是：

1. 依次测试 `10, 20, ..., 100` 个 PCA 主成分。
2. 对训练集执行 `fit_transform`。
3. 对测试集执行 `transform`。
4. 用降维后的训练集训练 5-NN。
5. 用降维后的测试集预测类别。
6. 计算 accuracy。
7. 如果当前 accuracy 更高，就更新最佳结果。

这里必须注意：

- 训练集用 `fit_transform`，因为 PCA 的方向只能从训练集学习。
- 测试集用 `transform`，不能重新 `fit`，否则会引入测试集信息。

当前 notebook 保存的输出显示：

```text
Best PCA Accuracy: 0.86 with 40 components.
```

#### 2. LLE 参数搜索

```python
best_lle_acc = 0
best_lle_params = (0, 0)

for n_neighbors in range(5, 51, 5):
    for n_components in range(5, 51, 5):
        try:
            lle = LocallyLinearEmbedding(
                n_neighbors=n_neighbors,
                n_components=n_components,
                method='standard'
            )
            X_train_lle = lle.fit_transform(X_train)
            X_test_lle = lle.transform(X_test)

            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(X_train_lle, y_train)
            y_pred_lle = knn.predict(X_test_lle)
            acc = accuracy_score(y_test, y_pred_lle)

            if acc > best_lle_acc:
                best_lle_acc = acc
                best_lle_params = (n_neighbors, n_components)

        except Exception as e:
            print(f"Failed for n_neighbors={n_neighbors}, n_components={n_components}: {e}")
```

这段代码用于搜索 LLE 的最佳参数组合。

LLE 有两个重要参数：

- `n_neighbors`：每个样本用多少个近邻来描述局部结构。
- `n_components`：降维后的维度。

这里测试：

- 邻居数：`5, 10, ..., 50`
- 组件数：`5, 10, ..., 50`

每组参数的流程是：

1. 创建 LLE 模型。
2. 对训练集执行 `fit_transform`。
3. 对测试集执行 `transform`。
4. 用 LLE 降维后的训练集训练 5-NN。
5. 在测试集上预测。
6. 计算 accuracy。
7. 如果效果更好，就保存当前参数。

`try-except` 的作用是防止某些 LLE 参数组合因为数值问题或约束问题失败。如果某组参数失败，程序会打印错误并继续尝试下一组。

当前 notebook 保存的输出显示：

```text
Best LLE Accuracy: 0.93 with 40 neighbors and 45 components.
```

#### 3. PCA 降维耗时

```python
start_time = time.time()
pca = PCA(n_components=best_pca_components, random_state=42)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
pca_time = time.time() - start_time
print(f"PCA Dimensionality Reduction Time: {pca_time:.2f} seconds")
```

这段代码使用前面搜索到的最佳 PCA 组件数重新执行一次降维，并记录耗时。

计时范围包括：

- 创建 PCA 模型。
- 在训练集上拟合并降维。
- 用同一个 PCA 模型转换测试集。

得到的 `X_train_pca` 和 `X_test_pca` 会用于后续 k-NN 分类。

#### 4. LLE 降维耗时

```python
start_time = time.time()
lle = LocallyLinearEmbedding(
    n_neighbors=best_lle_params[0],
    n_components=best_lle_params[1],
    method='standard'
)
X_train_lle = lle.fit_transform(X_train)
X_test_lle = lle.transform(X_test)
lle_time = time.time() - start_time
print(f"LLE Dimensionality Reduction Time: {lle_time:.2f} seconds")
```

这段代码使用前面搜索到的最佳 LLE 参数重新执行一次降维，并记录耗时。

`best_lle_params[0]` 是最佳邻居数，`best_lle_params[1]` 是最佳组件数。

通常 LLE 会比 PCA 更慢，因为 LLE 需要计算局部邻居关系和嵌入结构，而 PCA 主要是线性代数分解。

#### 5. 原始数据上的 k-NN

```python
start_time = time.time()
knn_raw = KNeighborsClassifier(n_neighbors=5)
knn_raw.fit(X_train, y_train)
y_pred_raw = knn_raw.predict(X_test)
knn_raw_time = time.time() - start_time
acc_raw = accuracy_score(y_test, y_pred_raw)
```

这段代码在原始 4096 维数据上训练和测试 k-NN。

它的作用是提供 baseline：

- 没有降维。
- 直接用原始像素特征分类。
- 记录 k-NN 的训练和预测总耗时。
- 计算 accuracy。

#### 6. PCA 降维数据上的 k-NN

```python
start_time = time.time()
knn_pca = KNeighborsClassifier(n_neighbors=5)
knn_pca.fit(X_train_pca, y_train)
y_pred_pca = knn_pca.predict(X_test_pca)
knn_pca_time = time.time() - start_time
acc_pca = accuracy_score(y_test, y_pred_pca)
```

这段代码在 PCA 降维后的数据上训练和测试 k-NN。

作用是观察 PCA 是否能：

- 降低维度。
- 减少 k-NN 计算时间。
- 尽量保持分类 accuracy。

因为 k-NN 在高维空间中距离计算成本较高，所以降维通常能加快它的预测过程。

#### 7. LLE 降维数据上的 k-NN

```python
start_time = time.time()
knn_lle = KNeighborsClassifier(n_neighbors=5)
knn_lle.fit(X_train_lle, y_train)
y_pred_lle = knn_lle.predict(X_test_lle)
knn_lle_time = time.time() - start_time
acc_lle = accuracy_score(y_test, y_pred_lle)
```

这段代码在 LLE 降维后的数据上训练和测试 k-NN。

作用是观察 LLE 是否能：

- 保留人脸数据的局部流形结构。
- 在较低维度中提升或保持 k-NN 分类效果。
- 与 PCA 和原始数据进行 accuracy、耗时对比。

当前 notebook 保存的输出中，LLE 的 k-NN accuracy 为 `0.93`，高于原始数据和 PCA 降维数据。

## 3. 两个实验的共同思路

这两份 notebook 都是在做机器学习中的一个典型流程：

1. 读取数据。
2. 观察数据。
3. 选择模型或算法。
4. 训练或拟合模型。
5. 生成预测或转换结果。
6. 与真实标签或评价指标对比。
7. 可视化结果。

区别在于：

- `KMeans.ipynb` 重点是无监督聚类，模型本身不使用真实标签。
- `LLEvsPCA.ipynb` 重点是降维与分类结合，用 k-NN 的分类 accuracy 来评价降维后的特征质量。

## 4. 关键概念总结

### K-Means

K-Means 会把数据分成若干 cluster，每个 cluster 有一个中心点。它只根据特征之间的距离聚类，不知道真实标签。

### PCA

PCA 是线性降维方法。它寻找数据中方差最大的方向，把高维数据投影到这些方向上。

优点是速度快、稳定、可解释性较强。

### LLE

LLE 是非线性降维方法。它假设数据位于一个低维流形上，通过保持样本和局部邻居之间的关系来降维。

优点是可能更好地保留复杂非线性结构，但通常比 PCA 更慢，并且对参数更敏感。

### k-NN

k-NN 根据测试样本附近的训练样本类别进行分类。它不需要显式训练复杂模型，但预测时需要计算距离，因此维度越高通常越慢。
