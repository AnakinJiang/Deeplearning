# 1、概述

Pytorch像是Numpy的替代物，语法类似。

# 2、数据类型

```python
import torch
import numpy as np
# Pytorch有不同类型的Tensor
# 32位浮点数 torch.FloatTensor（默认）
# 64位浮点数 torch.DoubleTensor
# 16位整型 torch.ShortTensor
# 32位整型 torch.intTensor
# 64位整型 torch.LongTensor
a = torch.Tensor([[2, 3], [4, 8], [7, 9]])
b = torch.LongTensor([[2, 3], [4, 8], [7, 9]])
c = torch.zeros((3, 2))
d = torch.randn((3, 2))
a[0, 1] = 100
numpy_b = b.numpy()
e = np.array([[2, 3], [4, 5]])
torch_e = torch.from_numpy(e)
f_torch_e = torch_e.float()
```

# 3、运算

```python
torch.abs(a) #取绝对值，输入参数为Tensor
torch.add(a,b) #求和，输入参数可以两个Tensor或者Tensor和标量
torch.clamp(a,l,u) #将Teonsor数据裁剪，a为被裁剪Tensor，l为下边界，u为上边界
# 如torch.clamp(a, -0.1, 0.1)表示上下界为-0.1，0.1
torch.div(a,b) #除法，可以两个Tensor也可以Tensor和标量
torch.mul(a,b) #乘法，可以两个Tensor也可以Tensor和标量
torch.pow(a,b) #乘方，可以两个Tensor也可以Tensor和标量
torch.mm(a,b) #乘法，严格要求矩阵乘矩阵
torch.mv(a,b) #乘法，严格要求矩阵乘向量
```

# 4、

