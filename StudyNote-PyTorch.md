> 这是 [动手学深度学习(PyTorch版)](https://github.com/ShusenTang/Dive-into-DL-PyTorch)的读书笔记. 后续可能会因个人懒的原因加入更多的DL笔记啥的..

```bash
#　给记性不好的我的Anaconda handbook
# 查看帮助
conda -h 
# 基于python3.6版本创建一个名字为python36的环境
conda create --name python36 python=3.6 
# 激活此环境
activate python36  
source activate python36 # linux/mac
# 再来检查python版本，显示是 3.6
python -V  
# 退出当前环境
deactivate python36 
# 删除该环境
conda remove -n python36 --all
# 或者 
conda env remove  -n python36

# 查看所以安装的环境
conda env list
```

# 动手学DL(PyTorch版)

## 1. 简介
- "用数据编程"
与其枯坐在房间里思考怎么设计一个识别猫的程序，不如收集一些已知包含猫与不包含猫的真实图像，然后我们的目标就转化成如何从这些图像入手得到一个可以推断出图像中是否有猫的函数。

- 算力增长 >> 存储增长
存储容量没能跟上数据量增长的步伐。与此同时，计算力的增长又盖过了数据量的增长。这样的趋势使得统计模型可以在优化参数上投入更多的计算力，但同时需要提高存储的利用效率，例如使用非线性处理单元。

- DL特点: 端到端
并不是将单独调试的部分拼凑起来组成一个系统，而是将整个系统组建好之后一起训练。

## 2. PyTorch基础操作
在PyTorch中，torch.Tensor是存储和变换数据的主要工具。Tensor近似与多维数组，具有自动求梯度与GPU计算功能.

- Tensor
  创建Tensor
  ```python
  import torch
  # 创建一个5 * 3未初始化的Tensor
  x = torch.empty(5, 3) 

  # 创建一个5 * 3 全为long 0的Tensor
  x = torch.zeros(5, 3, dtype=torch.long)

  # 输出形状
  print(x.size())
  print(x.shape)
  ```
  运算Tensor
  ```python
  # 加
  z = x + y

  torch.add(x, y, out=z)

  y.add_(x) # y += x
  ```

  索引(浅拷贝)
  ```python
  y = x[0, :]
  y += 1
  print(y)
  print(x[0, :]) # 源tensor也被改了
  ```
  改变形状
  `y = x.view(-1, 5)`

  clone(深拷贝)
  `x_cp = x.clone().view(15)`

   Tensor => Numpy
   `np = torch.ones(5).numpy()`
   Numpy => Tensor
   `ts = torch.from_numpy(np.ones(5))`

  Tensor with GPU
  ```python
  # Below codes CAN ONLY run with PyTorch-GPU
  if torch.cuda.is_available():
    gpu = torch.device("cuda")
    y = torch.ones_like(x, device=gpu) # y is in gpu
    x = x.to(gpu)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
  ```

- 梯度 (gradient)
  Tensor是这个包的核心类，如果将其属性`.requires_grad`设置为`True`，它将开始追踪(track)在其上的所有操作（这样就可以利用链式法则进行梯度传播了）。完成计算后，可以调用`.backward()`来完成所有梯度计算。此`Tensor`的梯度将累积到`.grad`属性中。`.detach()`可以阻止追踪
  e.g.
  创建一个Tensor并设置requires_grad=True:
  ```python
  x = torch.ones(2, 2, requires_grad=True)
  print(x)
  print(x.grad_fn)
  ```
  做运算操作
  ```python
  y = x + 2
  print(y)
  print(y.grad_fn)
  ```
  > 注意x是直接创建的，所以它没有`grad_fn`, 而y是通过一个加法操作创建的，所以它有一个为\<AddBackward>的`grad_fn`

  像x这种直接创建的被称为叶子节点, 叶子节点的`grad_fn`为`None`
  ```python
  print(x.is_leaf) # True
  print(y.is_leaf) # False
  ```

  通过`.requires_grad_()`来改变requires_grad属性
  ```python
  a = torch.randn(2, 2) # Default requires_grad = False
  print(a.requires_grad) # False
  a.requires_grad_(True)
  print(a.requires_grad) # True
  ```

  backward()自动计算梯度(grad)
  ```python
  x = torch.ones(2, 2, requires_grad=True) # x: [ [1, 1], [1, 1]]
  y = x + 2  # y: [ [3, 3], [3, 3]]
  z = y**2 * 3 # z: [ [27, 27], [27, 27]]
  z_mean = z.mean() # z_mean: [27]
  # z_mean 为一个标量, 所以无需指定求导变量
  z_mean.backward() # <=> out.backward(torch.tensor(1.))
  print(x.grad) # [ [4.5, 4.5], [4.5, 4.5]]
  ```

  如果我们想要修改`tensor`的数值，但是又不希望被`autograd`记录（即不会影响反向传播），那么我么可以对`tensor.data`进行操作。
  ```python
  x = torch.ones(1, requires_grad=True)

  print(x.data) # x.data is a tensor [1]
  print(x.data.requires_grad) # False

  y = x * 2
  x.data *= 100 # Only changes x's value, DO NOT backward.

  y.backward()
  print(x) # x: [100]
  print(x.grad) # 2
  ```

## 3. DL基础
### 线性回归 (Linear Regression)
- 线性回归的基本要素
  - 模型
  设房屋的面积为$x1$, 房龄为$x2$, 售出面积为$y$, 我们需要建立$y$关于$x1$, $x2$的表达式, 也即是模型 (model)
  - 训练数据
  通过数据来寻找特定的模型参数值，使模型在数据上的误差尽可能小。这个过程叫作模型训练 (model training)
  - 损失函数
  在机器学习里，将衡量误差的函数称为损失函数 (loss function)
  - 优化算法
  当模型和损失函数形式较为简单时，上面的误差最小化问题的解可以直接用公式表达出来。这类解叫作解析解（analytical solution）。
  线性回归和平方误差刚好属于这个范畴。然而，大多数深度学习模型并没有解析解，只能通过优化算法有限次迭代模型参数来尽可能降低损失函数的值。这类解叫作数值解（numerical solution）。

- 线性回归的表示方法
  ![线性回归神经网络图](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.1_linreg.svg)
  如图, 线性回归是个单层神经网络, 在线性回归中, $o$的计算完全依赖于$x1$和$x2$. 
  所以, 这里的输出层又叫全连接层 (fully-connected layer)或稠密层 (dense layer)

- 从0开始手撸线性回归
  即只用`Tensor`和`autograd`实现线性回归的训练
  见[`Source Code: LinearRegressTest.py`](PyTorch/LinearRegressTest.py)

- 线性回归简洁实现
  见[`Source Code: LinearRegressTestSimple.py`](PyTorch/LinearRegressTestSimple.py)

  > ***总结***
  **一般过程**:
  `准备数据集`->`读入数据` -> `定义模型` -> `初始化模型参数` -> `定义损失函数` -> `定义优化算法` -> `训练`
  **PyTorch框架对应**:
  `torch.utils.data`: 数据处理相关
  `torch.nn`: 神经网络的层
  `torch.nn.init`: 模块的初始化方法
  `torch.optim`: 优化算法

### Softmax回归
- 简介
  softmax回归同线性回归一样是个单层神经网络, 由于输出$o_1, o_2, o_3$的计算依赖于所有输入$x_1, x_2, x_3, x_4$, 故softmax回归的输出层也是个全连接层.
  softmax回归也将输入特征与权重做线性叠加. 不同点在于, softmax回归的输出值个数等于标签中的类别数.
  ![softmax回归神经网络图](https://tangshusen.me/Dive-into-DL-PyTorch/img/chapter03/3.4_softmaxreg.svg)

  > e.g.
  $o_1 = x_1w_{11} + x_2w_{21} + x_3w_{31} + x_4w{41} + b_1, $
  $o_2 = x_1w_{12} + x_2w_{22} + x_3w_{32} + x_4w{42} + b_2, $
  $o_3 = x_1w_{13} + x_2w_{23} + x_3w_{33} + x_4w{43} + b_3. $

  softmax运算符 (softmax operator)将输出层的输出值变为值为正且和为1的概率分布:
  $$y'_1, y'_2, y'_3 = softmax(o_1, o_2, o_3)$$
  其中
  $$y'_1 = { {exp(o_1)} \over {\sum^3_{i=1} exp(o_i)} },  
    y'_2 = { {exp(o_2)} \over {\sum^3_{i=1} exp(o_i)} }, 
    y'_3 = { {exp(o_3)} \over {\sum^3_{i=1} exp(o_i)} }
  $$






















