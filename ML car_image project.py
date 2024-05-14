import pandas as pd
import numpy as np
import time
import torch
from torch import nn
from torchvision.io import read_image
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from joblib import Parallel, delayed

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, labelfile, filepath, transform):
        super().__init__()
        self.labelfile = labelfile
        self.filepath = filepath
        self.transform = transform

    def __len__(self):
        return len(self.labelfile)

    def __getitem__(self, index):
        label = self.labelfile['Fraud'].iloc[index]
        figurename = self.labelfile['Name'].iloc[index]
        X = self.transform(read_image(self.filepath + "//" + figurename)).to(device)
        y = torch.tensor(label, dtype=torch.float32).reshape(1).to(device)
        return X, y

# 图像预处理
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 数据加载
data = pd.read_csv(r".\labeledfile_train.csv")
train, test = train_test_split(data, train_size=0.7)
traindata, validatedata = train_test_split(train, train_size=0.8)
fp = r".\大作业数据tostudents"

image_train = ImageDataset(traindata, fp, transform)
image_validate = ImageDataset(validatedata, fp, transform)
image_test = ImageDataset(test, fp, transform)

# 打印数据集信息
print("Train size:", len(image_train))
print("Validation size:", len(image_validate))
print("Test size:", len(image_test))

# 计算权重用于不平衡数据处理
targets = traindata['Fraud']
count_0 = len(targets[targets == 0])
count_1 = len(targets[targets == 1])
weight_for_0 = 1.0 / count_0
weight_for_1 = 1.0 / count_1
weights = [weight_for_0 if t == 0 else weight_for_1 for t in targets]
sampler = WeightedRandomSampler(weights, len(weights))

# 加载数据
load_train = DataLoader(image_train, batch_size=32, sampler=sampler)
load_validate = DataLoader(image_validate, batch_size=len(image_validate))
load_test = DataLoader(image_test, batch_size=len(image_test))

# 实例化模型
weights = ViT_B_16_Weights.DEFAULT
model = vit_b_16(weights=weights).to(device)

# 修改模型结构
for param in model.parameters():
    param.requires_grad = False

model.heads = nn.Sequential(
    nn.Linear(768, 256),
    nn.ReLU(),
    nn.Dropout(0.3062049759156305),
    nn.Linear(256, 1)
).to(device)

# 定义损失函数和优化器
bceloss = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=6.849171918430882e-05)

# 准备验证集和测试集
X_validate, y_validate = next(iter(load_validate))
X_test, y_test = next(iter(load_test))
y_validate = y_validate.cpu().detach().numpy()
y_test = y_test.cpu().detach().numpy()

# 定义动态阈值优化函数
def optimize_threshold(p_validate, y_validate, n_iterations=100):
    def train_classifier(l):
        def cut(x):
            cut_line = 0.01 * l
            return x > cut_line

        p_validate_cutted = cut(p_validate)
        f1 = f1_score(y_validate, p_validate_cutted, average="macro")
        return f1, 0.01 * l

    results = Parallel(n_jobs=-1)(delayed(train_classifier)(l) for l in range(n_iterations))
    results.sort(key=lambda x: x[0], reverse=True)
    best_f1, best_threshold = results[0] if results else (None, None)
    return best_f1, best_threshold

# 训练和验证
starttime = time.time()
trainingresult = []
steps = 0
best_threshold = 0
best_f1_on_validate = 0
for epoch in range(1, 11):
    epoch_loss = 0.0
    for X, y in load_train:
        X, y = X.to(device), y.to(device)
        model.train()
        steps += 1
        loss = bceloss(model(X), y)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if steps % 10 == 0:
            with torch.no_grad():
                model.eval()
                p_validate = nn.Sigmoid()(model(X_validate.to(device))).detach().cpu().numpy()
                current_best_f1, current_best_threshold = optimize_threshold(p_validate, y_validate)
                if current_best_f1 > best_f1_on_validate:
                    best_f1_on_validate = current_best_f1
                    best_threshold = current_best_threshold

                p_validate_binary = (p_validate > best_threshold).astype(int)
                validate_f1 = f1_score(y_validate, p_validate_binary, average="macro")
                time_elapse = round((time.time() - starttime) / 60, 1)
                trainingresult.append((steps, validate_f1, best_threshold))
                torch.save(model.state_dict(), str(steps) + ".model")
                print("Steps:", steps, "| Threshold:", best_threshold, "| F1:", validate_f1, "| Time:", time_elapse, "minutes")
    print("Epoch:", epoch, "| Epoch Loss:", epoch_loss, "\n" + "-"*100)

# 测试
index = np.argmax([res[1] for res in trainingresult])
optimum_steps = trainingresult[index][0]
optimum_threshold = trainingresult[index][2]
network_best = model
network_best.load_state_dict(torch.load(str(optimum_steps) + ".model"))  # 加载最优状态
network_best.eval()
p_test_predict = nn.Sigmoid()(network_best(X_test.to(device))).detach().cpu().numpy()
print("best model=", str(optimum_steps))
print("best threshold=", optimum_threshold)
print("F1 in the test sample =", f1_score(y_test, (p_test_predict > optimum_threshold).astype(int), average="macro"))
