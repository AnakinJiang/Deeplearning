# 安装

```shell
pip install keras
pip install --upgrade keras
```

# 概述

Keras基于两个Backend，一个是 Theano，一个是 Tensorflow。如果我们选择Theano作为Keras的Backend， 那么Keras就用 Theano 在底层搭建你需要的神经网络；Tensorflow同理。可以指定使用哪个Backend：

如果要在代码里指定，可以使用如下格式：

```python
import keras #每次当我们import keras的时候，就会看到屏幕显示当前使用的 Backend
Using Theano Backend
```

另外，我们也可以在文件中修改全局配置，找到这个文件~/.keras/keras.json，直接更改

```json
{
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "image_data_format": "channels_last", 
    "backend": "tensorflow"
}
```

也可以在terminal中指定

```shell
KERAS_BACKEND=tensorflow python -c "from keras import backend"
```

也能在代码中指定环境变量

```python
import os
os.environ['KERAS_BACKEND']='theano'
```

