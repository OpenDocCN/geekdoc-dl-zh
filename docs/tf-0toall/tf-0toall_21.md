# 张量操作实验 08

# 张量操作实验 08

```
# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint
tf.set_random_seed(777)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
sess = tf.InteractiveSession() 
```

## 简单数组

```
t = np.array([0., 1., 2., 3., 4., 5., 6.])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:]) 
```

```
array([ 0.,  1.,  2.,  3.,  4.,  5.,  6.])
(7,) 1
0.0 1.0 6.0
[ 2\.  3\.  4.] [ 4\.  5.]
[ 0\.  1.] [ 3\.  4\.  5\.  6.] 
```

# 二维数组

```
t = np.array([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.], [10., 11., 12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape 
```

```
array([[  1.,   2.,   3.],
       [  4.,   5.,   6.],
       [  7.,   8.,   9.],
       [ 10.,  11.,  12.]])
2
(4, 3) 
```

## 形状、秩、轴

```
t = tf.constant([1,2,3,4])
tf.shape(t).eval() 
```

```
array([4], dtype=int32) 
```

```
t = tf.constant([[1,2],
                 [3,4]])
tf.shape(t).eval() 
```

```
array([2, 2], dtype=int32) 
```

```
t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],[[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval() 
```

```
array([1, 2, 3, 4], dtype=int32) 
```

```
[
    [
        [
            [1,2,3,4], 
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20], 
            [21,22,23,24]
        ]
    ]
] 
```

```
[[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]] 
```

## 矩阵乘法 VS 逐元素相乘

```
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
tf.matmul(matrix1, matrix2).eval() 
```

```
array([[ 12.]], dtype=float32) 
```

```
(matrix1*matrix2).eval() 
```

```
array([[ 6.,  6.],
       [ 6.,  6.]], dtype=float32) 
```

## 注意广播

```
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
(matrix1+matrix2).eval() 
```

```
array([[ 5.,  5.],
       [ 5.,  5.]], dtype=float32) 
```

```
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
(matrix1+matrix2).eval() 
```

```
array([[ 5.,  5.]], dtype=float32) 
```

## 变量初始化的随机值

```
tf.random_normal([3]).eval() 
```

```
array([ 2.20866942, -0.73225045,  0.33533147], dtype=float32) 
```

```
tf.random_uniform([2]).eval() 
```

```
array([ 0.08186948,  0.42999184], dtype=float32) 
```

```
tf.random_uniform([2, 3]).eval() 
```

```
array([[ 0.43535876,  0.76933432,  0.65130949],
       [ 0.90863407,  0.06278825,  0.85073185]], dtype=float32) 
```

## 求均值/和

```
tf.reduce_mean([1, 2], axis=0).eval() 
```

```
1 
```

```
x = [[1., 2.],
     [3., 4.]]

tf.reduce_mean(x).eval() 
```

```
2.5 
```

```
tf.reduce_mean(x, axis=0).eval() 
```

```
array([ 2.,  3.], dtype=float32) 
```

```
tf.reduce_mean(x, axis=1).eval() 
```

```
array([ 1.5,  3.5], dtype=float32) 
```

```
tf.reduce_mean(x, axis=-1).eval() 
```

```
array([ 1.5,  3.5], dtype=float32) 
```

```
tf.reduce_sum(x).eval() 
```

```
10.0 
```

```
tf.reduce_sum(x, axis=0).eval() 
```

```
array([ 4.,  6.], dtype=float32) 
```

```
tf.reduce_sum(x, axis=-1).eval() 
```

```
array([ 3.,  7.], dtype=float32) 
```

```
tf.reduce_mean(tf.reduce_sum(x, axis=-1)).eval() 
```

```
5.0 
```

## 带轴的 Argmax

```
x = [[0, 1, 2],
     [2, 1, 0]]
tf.argmax(x, axis=0).eval() 
```

```
array([1, 0, 0]) 
```

```
tf.argmax(x, axis=1).eval() 
```

```
array([2, 0]) 
```

```
tf.argmax(x, axis=-1).eval() 
```

```
array([2, 0]) 
```

## 重塑、挤压、扩展维度

```
t = np.array([[[0, 1, 2], 
               [3, 4, 5]],

              [[6, 7, 8], 
               [9, 10, 11]]])
t.shape 
```

```
(2, 2, 3) 
```

```
tf.reshape(t, shape=[-1, 3]).eval() 
```

```
array([[ 0,  1,  2],
       [ 3,  4,  5],
       [ 6,  7,  8],
       [ 9, 10, 11]]) 
```

```
tf.reshape(t, shape=[-1, 1, 3]).eval() 
```

```
array([[[ 0,  1,  2]],

       [[ 3,  4,  5]],

       [[ 6,  7,  8]],

       [[ 9, 10, 11]]]) 
```

```
tf.squeeze([[0], [1], [2]]).eval() 
```

```
array([0, 1, 2], dtype=int32) 
```

```
tf.expand_dims([0, 1, 2], 1).eval() 
```

```
array([[0],
       [1],
       [2]], dtype=int32) 
```

## 独热编码

```
tf.one_hot([[0], [1], [2], [0]], depth=3).eval() 
```

```
array([[[ 1.,  0.,  0.]],

       [[ 0.,  1.,  0.]],

       [[ 0.,  0.,  1.]],

       [[ 1.,  0.,  0.]]], dtype=float32) 
```

```
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3]).eval() 
```

```
array([[ 1.,  0.,  0.],
       [ 0.,  1.,  0.],
       [ 0.,  0.,  1.],
       [ 1.,  0.,  0.]], dtype=float32) 
```

## 类型转换

```
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval() 
```

```
array([1, 2, 3, 4], dtype=int32) 
```

```
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval() 
```

```
array([1, 0, 1, 0], dtype=int32) 
```

## 堆叠

```
x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
tf.stack([x, y, z]).eval() 
```

```
array([[1, 4],
       [2, 5],
       [3, 6]], dtype=int32) 
```

```
tf.stack([x, y, z], axis=1).eval() 
```

```
array([[1, 2, 3],
       [4, 5, 6]], dtype=int32) 
```

## Ones 和 Zeros 类似

```
x = [[0, 1, 2],
     [2, 1, 0]]

tf.ones_like(x).eval() 
```

```
array([[1, 1, 1],
       [1, 1, 1]], dtype=int32) 
```

```
tf.zeros_like(x).eval() 
```

```
array([[0, 0, 0],
       [0, 0, 0]], dtype=int32) 
```

## Zip

```
for x, y in zip([1, 2, 3], [4, 5, 6]):
    print(x, y) 
```

```
1 4
2 5
3 6 
```

```
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z) 
```

```
1 4 7
2 5 8
3 6 9 
```

## 转置

```
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t) 
```

```
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]]) 
```

```
t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(sess.run(t1).shape)
pp.pprint(sess.run(t1)) 
```

```
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 6,  7,  8]],

       [[ 3,  4,  5],
        [ 9, 10, 11]]]) 
```

```
t = tf.transpose(t1, [1, 0, 2])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t)) 
```

```
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]]) 
```

```
t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(sess.run(t2).shape)
pp.pprint(sess.run(t2)) 
```

```
(2, 3, 2)
array([[[ 0,  6],
        [ 1,  7],
        [ 2,  8]],

       [[ 3,  9],
        [ 4, 10],
        [ 5, 11]]]) 
```

```
t = tf.transpose(t2, [2, 0, 1])
pp.pprint(sess.run(t).shape)
pp.pprint(sess.run(t)) 
```

```
(2, 2, 3)
array([[[ 0,  1,  2],
        [ 3,  4,  5]],

       [[ 6,  7,  8],
        [ 9, 10, 11]]]) 
```
