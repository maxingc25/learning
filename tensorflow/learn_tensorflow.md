```python
import numpy as np
```


```python
## 检查版本
import tensorflow as tf
print('tf.__version__=', tf.__version__)
```

    d:\anaconda\lib\site-packages\scipy\__init__.py:146: UserWarning: A NumPy version >=1.16.5 and <1.23.0 is required for this version of SciPy (detected version 1.26.1
      warnings.warn(f"A NumPy version >={np_minversion} and <{np_maxversion}"
    

    tf.__version__= 2.14.0
    

# 1. 数据类型

## 1.1 张量的基础类型


```python
print(tf.constant(True))
print(tf.constant(1))
print(tf.constant(1.0))
print(tf.constant(1.0, dtype=tf.float64))
print(tf.constant('hello'))

## 通过numpy构建，与上面的float情况表现不一致
print(tf.constant(np.array(2.0)))
print(tf.constant(np.array(2.0), dtype=tf.float32))
```

    tf.Tensor(True, shape=(), dtype=bool)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(1.0, shape=(), dtype=float32)
    tf.Tensor(1.0, shape=(), dtype=float64)
    tf.Tensor(b'hello', shape=(), dtype=string)
    tf.Tensor(2.0, shape=(), dtype=float64)
    tf.Tensor(2.0, shape=(), dtype=float32)
    

numpy创建数组时默认的浮点型是double，而tf在创建张量时默认是float32。tf在接受numpy数组时会同时接受其元素的数据类型。

## 1.2 Variable是特殊的张量
Variable也是一种tensor，一般用于模型中需要再训练过程中进行调整的参数


```python
print(tf.Variable(1.0))
print(isinstance(tf.Variable(1.0), tf.Tensor))
print(type(tf.Variable(1.0)))
print(tf.is_tensor(tf.Variable(1.0)))
```

    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=1.0>
    False
    <class 'tensorflow.python.ops.resource_variable_ops.ResourceVariable'>
    True
    


```python
## 特殊性
a = tf.Variable(1.0)
a.assign(2.0)
print(a)
print(a.trainable)

b=tf.constant(1.0)
try:
    print(b.trainable)
    b.assign(2.0)
except Exception as e:
    print('error:', e)
```

    <tf.Variable 'Variable:0' shape=() dtype=float32, numpy=2.0>
    True
    error: 'tensorflow.python.framework.ops.EagerTensor' object has no attribute 'trainable'
    

Variable在初始化之后还是可以修改其内容的，并且其包含了“trainable”属性。该属性在训练时，可以用于判断这个Variable是否需要被调整。而一般的Tensor在初始化之后是不能修改内容的。如果需要修改，就只能生成一个新的Tensor。并且一般的Tensor没有“trainable”属性，说明在训练时，是完全不需要在误差反传中进行调整。

## 1.3 张量的常用属性


```python
a = tf.constant([1.0])
print('张量的形状：', a.shape)
print('张量的维度：', a.ndim)
print('张量元素的类型：', a.dtype)
print('张量转换为numpy数组：', a.numpy())
```

    张量的形状： (1,)
    张量的维度： 1
    张量元素的类型： <dtype: 'float32'>
    张量转换为numpy数组： [1.]
    

# 2. 创建张量


```python
# 从python基础类型创建
print(tf.constant(1))
print(tf.convert_to_tensor(1))

# 从list创建 
print(tf.constant([1,2,3]))

# 从numpy数组创建
print(tf.constant(np.array([1,2,3])))

#使用tf中类似numpy的操作创建
## 特殊值数组: ones, zeros, eye
print(tf.ones(shape=(1,3), dtype=tf.float32))
## 线性划分数组
print(tf.linspace(0.0,1.0,3))
## 同形状填充
print(tf.ones_like(tf.linspace(0.0,1,0,4)))
## 通用数值填充
print(tf.fill((1,5), 1.0))

#随机函数创建
print(tf.random.normal((1,2)))
print(tf.random.truncated_normal((1,2)))
print(tf.random.uniform((1,2), minval=0.0, maxval=1.0))

#运算过程中产生
a = tf.constant([[1,2], [3,4]])
b = tf.constant([[5,6], [7,8]])
print(a*b)
```

    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor([[1. 1. 1.]], shape=(1, 3), dtype=float32)
    tf.Tensor([0.  0.5 1. ], shape=(3,), dtype=float32)
    tf.Tensor([], shape=(0,), dtype=float32)
    tf.Tensor([[1. 1. 1. 1. 1.]], shape=(1, 5), dtype=float32)
    tf.Tensor([[-0.49769893 -1.3980428 ]], shape=(1, 2), dtype=float32)
    tf.Tensor([[ 0.5500834  -0.28357336]], shape=(1, 2), dtype=float32)
    tf.Tensor([[0.9248296  0.21447694]], shape=(1, 2), dtype=float32)
    tf.Tensor(
    [[ 5 12]
     [21 32]], shape=(2, 2), dtype=int32)
    

# 张量索引和切片


```python
# 基本索引
a = tf.constant([[1,2,3],[4,5,6]])
print(a[0])
print(a[0][0])

# 切片 类似numpy风格
print(a[0,1])        # 取元素
print(a[0:2, 0])     # 取第0列，省略step
print(a[0, 0:3:2])   # 完全形式的切片
print(a[0, ::-1])    # 省略start和end
print(a[0, :-3:-1])  # 省略start
print(a[0, -2::-1])  # 省略end
print(a[0, ...])     # 使用省略号, 省略号表示被省略的维度中取所有的shape
print(a[..., 1])     # 使用省略号

# 切片函数
b = tf.ones((2,3,4,5,6), dtype=tf.int32)
# 收集指定维度的若干个索引，其他维度保持形状不变
print(tf.gather(b, axis=4, indices=[2]).shape)
# 收集指定的若干个维度的索引
print(tf.gather_nd(b, [[0,0,0], [0,0,1]]).shape)
# 使用bool蒙版
mask = tf.constant([False, False, False, True])
print(tf.boolean_mask(b, mask=mask, axis=2).shape)
```

    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor(1, shape=(), dtype=int32)
    tf.Tensor(2, shape=(), dtype=int32)
    tf.Tensor([1 4], shape=(2,), dtype=int32)
    tf.Tensor([1 3], shape=(2,), dtype=int32)
    tf.Tensor([3 2 1], shape=(3,), dtype=int32)
    tf.Tensor([3 2], shape=(2,), dtype=int32)
    tf.Tensor([2 1], shape=(2,), dtype=int32)
    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    tf.Tensor([2 5], shape=(2,), dtype=int32)
    (2, 3, 4, 5, 1)
    (2, 5, 6)
    (2, 3, 1, 5, 6)
    


```python

```
