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
  # Below codes can only run with PyTorch-GPU
  if torch.cuda.isavailable():
    gpu = torch.device("cuda")
    y = torch.ones_like(x, device=gpu) # y is in gpu
    x = x.to(gpu)
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))
  ```




























