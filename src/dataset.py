"""
读取datadeal处理出来的json文件
定义图像处理的具体方式（分训练集和非训练集）
构造供dataloader使用的类
"""
import json
from PIL import Image
from pathlib import Path
from torchvision import transforms
from torch.utils.data import Dataset
IMG_SIZE = 224
META_PATH = Path("data/split_meta.json")

# 读json文件并转化为python对象
def load_meta() :
  with open(META_PATH, encoding = "utf-8") as f :
    return json.load(f)

# 图像处理（训练集要数据增强，测试集、验证集不用）
def build_transforms(meta, img_size = IMG_SIZE) :
  mean = meta["normalize"]["mean"]
  std = meta["normalize"]["std"]
  train_tf = transforms.Compose([
    # RandomResizedCrop：随机裁剪图像的一块区域，再缩放到指定大小
    transforms.RandomResizedCrop(img_size, scale = (0.7, 1.0)),
    # RandomHorizontalFlip：以一定概率给图像左右翻转，默认0.5
    transforms.RandomHorizontalFlip(),
    # ColorJitter：随机调整图像的亮度、对比度、饱和度和色调，在这里不能调色调，因为识别花卉色调很重要
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    # RandomErasing：随机选一块小的矩形区域擦除
    transforms.RandomErasing(p = 0.25, scale = (0.02, 0.33)),
    # 标准化
    transforms.Normalize(mean, std),
  ])
  eval_tf = transforms.Compose([
    transforms.Resize((img_size, img_size)), 
    transforms.ToTensor(), 
    transforms.Normalize(mean, std), 
  ])
  return train_tf, eval_tf

# 构建数据类
class FlowerDataset(Dataset) :
  def __init__(self, paths, labels, class_to_idx, transform) :
    self.paths = paths
    self.labels = [class_to_idx[name] for name in labels]
    self.transform = transform
  def __len__(self) :
    return len(self.paths)
  def __getitem__(self, i) :
    img = Image.open(self.paths[i]).convert("RGB")
    img = self.transform(img)
    return img, self.labels[i]
  