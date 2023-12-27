# %%
import numpy as np

# %%
## 检查版本
import tensorflow as tf
print('tf.__version__=', tf.__version__)

# %% [markdown]
# ## 张量的基础类型

# %%
print(tf.constant(True))
print(tf.constant(1))
print(tf.constant(1.0))
print(tf.constant(1.0, dtype=tf.float64))
print(tf.constant('hello'))

## 通过numpy构建，与上面的float情况表现不一致
print(tf.constant(np.array(2.0)))
print(tf.constant(np.array(2.0), dtype=tf.float32))

# %% [markdown]
# numpy创建数组时默认的浮点型是double，而tf在创建张量时默认是float32。tf在接受numpy数组时会同时接受其元素的数据类型。

# %%



