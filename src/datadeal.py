"""
预处理数据
"""
import json
import random
import hashlib
import torch as th
from pathlib import Path
from torchvision import transforms
from PIL import Image, UnidentifiedImageError, ImageOps
SEED = 42
rng = random.Random(SEED)
img_dir = Path("data/")
exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}

# 检测文件是否损坏
def is_image_valid(path) :
  try :
    with Image.open(path) as img :
      img.verify()
    with Image.open(path) as img :
      img.load()
    return True, None
  except(UnidentifiedImageError, OSError, SyntaxError, ValueError) as err :
    return False, f"{type(err).__name__}: {err}"

# 计算图片哈希值
def content_hash(path) :
  with Image.open(path) as im:
    im = ImageOps.exif_transpose(im).convert("RGB")
    h = hashlib.md5()
    h.update(str(im.size).encode())
    h.update(im.tobytes())
  return h.hexdigest()

# 数据收集（此时检测文件是否损坏+去重）以及按一定概率把数据划分进验证集
def collect_images(root, pv, hs) :
  root = Path(root)
  items = []
  items_val = []
  for cls_dir in sorted(root.iterdir()) :
    if not cls_dir.is_dir() :
      continue
    for p in cls_dir.rglob('*') :
      if p.suffix.lower() in exts :
        ok, err = is_image_valid(p)
        if ok :
          h = content_hash(p)
          if h not in hs :
            hs.add(h)
            rnd = rng.random()
            if rnd >= pv :
              items.append((p, cls_dir.name))
            else :
              items_val.append((p, cls_dir.name))
  return items, items_val

# 分别统计train和test的数据，（训练集分1/9出去到val）
hs_train = set()
hs_test = set()
items_train, items_val1 = collect_images(img_dir / "train" , 1 / 9, hs_train)
items_test, items_val2 = collect_images(img_dir / "test" , 0, hs_test)
items_val = items_val1 + items_val2
paths_train = [str(p) for p, _ in items_train]
labels_train = [l for _, l in items_train]
paths_test = [str(p) for p, _ in items_test]
labels_test = [l for _, l in items_test]
paths_val = [str(p) for p, _ in items_val]
labels_val = [l for _, l in items_val]

# 抽400张训练集的图片大致算一下训练集的平均数和标准差，dataset时要用
files = [Path(p) for p in paths_train]
random.Random(SEED).shuffle(files)
sample = files[:400]
to_tensor = transforms.ToTensor()
sum1 = th.zeros(3)
sum2 = th.zeros(3)
for p in sample :
  x = to_tensor(Image.open(p).convert("RGB"))
  sum1 += x.mean(dim = (1, 2))
  sum2 += (x ** 2).mean(dim = (1, 2))
n = len(sample)
MEAN = [round(v, 4) for v in (sum1 / n).tolist()]
STD = [round(v, 4) for v in (sum2 / n - (sum1 / n) ** 2).sqrt().tolist()]

# 把训练集、验证集、测试集存成一个json
json.dump({
  "class_to_idx" : {c: i for i, c in enumerate(sorted(set(labels_train)))}, 
  "normalize" : {"mean" : MEAN, "std" : STD, "n_sample": n}, 
  "train" : {"paths" : paths_train, "labels" : labels_train}, 
  "val" : {"paths": paths_val, "labels": labels_val}, 
  "test" : {"paths": paths_test, "labels": labels_test}, 
}, open("data/split_meta.json", "w", encoding = "utf-8"), ensure_ascii = False, indent = 2)

