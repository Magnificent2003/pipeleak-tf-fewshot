1.训练
python train.py --config configs/maml_resnet18.yaml
python train.py --config configs/maml_darknet19.yaml

2.数据格式

/dataset
dataset
│
├── class1
│     ├── 1.jpg
│     ├── 2.jpg
│
├── class2
│     ├── 1.jpg
│     ├── 2.jpg
│
├── class3
│
└── class4