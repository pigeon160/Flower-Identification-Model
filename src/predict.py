"""
用训练好的模型来预测
"""
import sys
sys.path.insert(0, "src")
import numpy as np
import torch as th
from torch.utils.data import DataLoader
from dataset import FlowerDataset, load_meta, build_transforms
from model import Mymodel
BATCH_SIZE = 64

# 载入模型
CKPT_PATH = "outputs/best_model.pth"
ckpt = th.load(CKPT_PATH, weights_only = False)
model = Mymodel()
model.load_state_dict(ckpt["state_dict"])
model.eval()

# 建立类别到数字及数字到类别的双向映射
class_to_idx = ckpt["class_to_idx"]
idx_to_class = {v : k for k, v in class_to_idx.items()}

# 加载测试集、创建数据加载器
meta = load_meta()
_, eval_tf = build_transforms(meta, img_size = ckpt["img_size"])
test_ds = FlowerDataset(meta["test"]["paths"], meta["test"]["labels"], class_to_idx, eval_tf)
test_loader = DataLoader(test_ds, batch_size = BATCH_SIZE, shuffle = False)

# 测试
preds = []
correct, total, loss_sum = 0, 0, 0.0
with th.no_grad() :
  for x, y in test_loader :
    logits = model(x)
    pred = logits.argmax(dim = 1)
    total += y.numel()
    correct += (pred == y).sum().item()
    loss_sum += model.criterion(logits, y).item() * y.numel()
    preds.append(pred.numpy())
  acc = correct / total
  loss = loss_sum / total
print(f"test_accuracy : {acc:.4%} test_loss : {loss:.4f}")

# 结果保存
preds = np.concatenate(preds)
p2name = [idx_to_class[i] for i in preds]
with open("outputs/predictions.csv", "w", encoding = "utf-8") as f :
  f.write("path,true_label,pred_label\n")
  for p, t, r in zip(meta["test"]["paths"], meta["test"]["labels"], p2name) :
    f.write(f"{p},{t},{r}\n")