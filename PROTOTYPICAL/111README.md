训练说明：
1.进行训练直接python train.py即可 如果要修改训练的轮数，backbone都在train.py里面修改
model = get_model("protonet_resnet18")
model = get_model("protonet_darknet19")
episodes = 10000
2.这个网络进行训练的时候不需要val test 
是要按照
data/mydataset
│
├── class1
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── ...
│
├── class2
│   ├── img1.jpg
│   ├── img2.jpg
│   └── ...
│
├── class3
│   ├── img1.jpg
│   └── ...
│
└── class4
这样子的格式进行训练。
参数有n_way = 4，n_support = 5，n_query = 15：
4 个类别（我们的类别数量）
每类 5 support
每类 15 query（这个5 15应该可以自己更改）
support：每个类别给模型 5 个已知样本；query：每类的测试样本。
训练的流程是
Step1 计算 prototype，对 support 特征求平均，得到 5 个 prototype；
Step2 计算距离，每个 query 样本计算到 prototype 的距离（一般是Euclidean distance）；
Step3 分类，选择最近的 prototype
所以这个方法的原理我感觉还是
先用深度神经网络进行特征提取，然后和传统的机器学习一样，计算特征之间的距离进行分类。
我训练了一下。其实效果也可以；看看后续如何说明我们的优势。


我已经把所有数据变成图像并进行分类的代码写好了
先运行build_datasets.py 再运行build_stft_datasets.py 最后运行classify_images.py
（如果报错 检查一下classify_images.py路径的问题）

或者就是到时候我直接把图像数据给你 

3.运行时间 
n_way = 4
n_support = 5
n_query = 15

episodes = 10000
这个设置 用组里面的服务器 2080  跑了几十分钟差不多
