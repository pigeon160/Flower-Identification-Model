"""
模型训练并存储、画图
"""
import random
import numpy as np
import torch as th
import matplotlib
matplotlib.use("Agg")      
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataset import FlowerDataset, load_meta, build_transforms
from model import Mymodel
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 80
LR = 1e-3
WEIGHT_DECAY = 1e-4

# 设置随机种子保证可复现
def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  th.manual_seed(seed)
set_seed(SEED)

# 读数据、加载出训练集和验证集
meta = load_meta()
c2i = meta["class_to_idx"]
train_tf, eval_tf = build_transforms(meta, img_size = IMG_SIZE)
train_ds = FlowerDataset(meta["train"]["paths"], meta["train"]["labels"], c2i, train_tf)
val_ds = FlowerDataset(meta["val"]["paths"], meta["val"]["labels"], c2i, eval_tf)

# 创建训练集和验证集的数据加载器
train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle = True)
val_loader = DataLoader(val_ds, batch_size = BATCH_SIZE, shuffle = False)

# 引入模型，优化器选择AdamW，调度器选择余弦退火
model = Mymodel()
optimizer = th.optim.AdamW(model.parameters(), lr = LR, weight_decay = WEIGHT_DECAY)
scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max = EPOCHS, eta_min = 1e-5)

# 开始训练 顺便把验证集正确率最高的模型存下来
best_acc = 0.0
accs_train = []
accs_val = []
losses_train = []
losses_val = []
for epoch in range(1, EPOCHS + 1) :
  model.train()
  train_loss, train_correct, train_total = 0.0, 0, 0
  for x, y in train_loader :
    optimizer.zero_grad()
    logits = model(x)
    loss = model.criterion(logits, y)
    loss.backward()
    optimizer.step()
    train_total += y.numel()
    train_loss += loss.item() * y.numel()
    train_correct += (logits.argmax(dim = 1) == y).sum().item()
  accs_train.append(train_correct / train_total)
  losses_train.append(train_loss / train_total)
  scheduler.step()
  model.eval()
  val_loss, val_correct, val_total = 0.0, 0, 0
  with th.no_grad() :
    for x, y in val_loader :
      logits = model(x)
      pred = logits.argmax(dim = 1)
      val_total += y.numel()
      val_correct += (pred == y).sum().item()
      val_loss += model.criterion(logits, y).item() * y.numel() 
  val_acc = val_correct / val_total
  accs_val.append(val_acc)
  losses_val.append(val_loss / val_total)
  if val_acc > best_acc :
    best_acc = val_acc
    th.save({
      "state_dict"   : model.state_dict(), 
      "class_to_idx" : c2i, 
      "img_size"     : IMG_SIZE, 
      "val_acc"      : val_acc, 
    }, "outputs/best_model.pth")
  print(f"epoch {epoch:2d} | train {accs_train[-1]:6.2%} loss {losses_train[-1]:.4f}"
        f" | val {val_acc:6.2%} loss {losses_val[-1]:.4f}")

# 作图
plt.figure(figsize = (6, 4))
plt.plot(range(1, EPOCHS + 1), accs_train, label = "train")
plt.plot(range(1, EPOCHS + 1), accs_val, label = "val")
plt.xlabel("epoch"); plt.ylabel("accuracy")
plt.title("Accuracy Trend"); plt.legend()
plt.savefig("docs/accuracy_curve.png", dpi = 150)
plt.close()

plt.figure(figsize = (6, 4))
plt.plot(range(1, EPOCHS + 1), losses_train, label = "train")
plt.plot(range(1, EPOCHS + 1), losses_val, label = "val")
plt.xlabel("epoch"); plt.ylabel("loss")
plt.title("Loss Curve"); plt.legend()
plt.savefig("docs/loss_curve.png", dpi = 150)