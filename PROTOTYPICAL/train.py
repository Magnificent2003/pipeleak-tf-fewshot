import torch
import torch.optim as optim
import matplotlib.pyplot as plt
import pandas as pd
import os

from protonets.models.factory import get_model
from protonets.data.custom_dataset import FewShotDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# dataset
dataset = FewShotDataset("data/mydataset")


# model
model = get_model("protonet_resnet18")
model = model.to(device)
model.train()


# optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)


# few-shot setting
n_way = 4
n_support = 5
n_query = 15

episodes = 10000


# ===== 保存训练历史 =====
loss_history = []
acc_history = []
episode_history = []


# ===== 训练 =====
for episode in range(episodes):

    sample = dataset.sample_episode(
        n_way,
        n_support,
        n_query
    )

    # move to gpu
    sample["xs"] = sample["xs"].to(device)
    sample["xq"] = sample["xq"].to(device)

    # forward
    loss, output = model.loss(sample)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 记录数据
    loss_val = output["loss"]
    acc_val = output["acc"]

    loss_history.append(loss_val)
    acc_history.append(acc_val)
    episode_history.append(episode)

    if episode % 50 == 0:
        print(
            "episode:", episode,
            "loss:", loss_val,
            "acc:", acc_val
        )


# ===== 保存模型 =====
torch.save(model.state_dict(), "protonet.pth")
print("模型已保存：protonet.pth")


# ===== 保存训练日志 =====
df = pd.DataFrame({
    "episode": episode_history,
    "loss": loss_history,
    "accuracy": acc_history
})

df.to_csv("training_log.csv", index=False)

print("训练日志保存：training_log.csv")


# ===== 可视化 Loss =====
plt.figure(figsize=(8,5))
plt.plot(loss_history)
plt.title("Training Loss")
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.grid(True)

plt.savefig("loss_curve.png", dpi=300)
plt.close()


# ===== 可视化 Accuracy =====
plt.figure(figsize=(8,5))
plt.plot(acc_history)
plt.title("Training Accuracy")
plt.xlabel("Episode")
plt.ylabel("Accuracy")
plt.grid(True)

plt.savefig("accuracy_curve.png", dpi=300)
plt.close()

print("可视化结果保存：loss_curve.png / accuracy_curve.png")