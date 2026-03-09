import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 设备设置
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------- 1. 神经网络定义 ----------------------
class PINN_Burgers(nn.Module):
    def __init__(self):
        super(PINN_Burgers, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 32), nn.Tanh(),
            nn.Linear(32, 1)
        )

    def forward(self, x, t):
        return self.net(torch.cat([x, t], dim=1))


# ---------------------- 2. 损失函数定义 ----------------------
def pde_loss(model, x, t, nu=0.01 / np.pi):
    x.requires_grad_(True)
    t.requires_grad_(True)
    u = model(x, t)

    # 自动微分
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    # Burgers 方程残差
    residual = u_t + u * u_x - nu * u_xx
    return torch.mean(residual ** 2)


def ic_loss(model, x):
    t0 = torch.zeros_like(x)
    u_pred = model(x, t0)
    u_true = -torch.sin(np.pi * x)
    return torch.mean((u_pred - u_true) ** 2)


def bc_loss(model, t):
    x0 = torch.zeros_like(t)
    x1 = torch.ones_like(t)
    u0 = model(x0, t)
    u1 = model(x1, t)
    return torch.mean(u0 ** 2) + torch.mean(u1 ** 2)


# ---------------------- 3. 数据采样 ----------------------
def sample_points(n_f, n_ic, n_bc):
    # 内部点（PDE残差）
    x_f = torch.rand(n_f, 1, device=device)
    t_f = torch.rand(n_f, 1, device=device)
    # 初始点
    x_ic = torch.rand(n_ic, 1, device=device)
    t_ic = torch.zeros_like(x_ic)
    # 边界点
    t_bc = torch.rand(n_bc, 1, device=device)
    x_bc0 = torch.zeros_like(t_bc)
    x_bc1 = torch.ones_like(t_bc)
    return (x_f, t_f), (x_ic, t_ic), (t_bc,)


# ---------------------- 4. 训练 ----------------------
model = PINN_Burgers().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
n_epochs = 5000
n_f, n_ic, n_bc = 2000, 200, 200

for epoch in range(n_epochs):
    optimizer.zero_grad()
    (x_f, t_f), (x_ic, t_ic), (t_bc,) = sample_points(n_f, n_ic, n_bc)

    loss_pde = pde_loss(model, x_f, t_f)
    loss_ic = ic_loss(model, x_ic)
    loss_bc = bc_loss(model, t_bc)
    total_loss = loss_pde + loss_ic + loss_bc

    total_loss.backward()
    optimizer.step()

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}, Loss: {total_loss.item():.4f}")

# ---------------------- 5. 可视化结果 ----------------------
x = np.linspace(0, 1, 100)
t = np.linspace(0, 1, 100)
X, T = np.meshgrid(x, t)
X_flat = torch.FloatTensor(X.flatten()[:, None]).to(device)
T_flat = torch.FloatTensor(T.flatten()[:, None]).to(device)

with torch.no_grad():
    U_pred = model(X_flat, T_flat).cpu().numpy().reshape(X.shape)

# 3D 曲面图
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X, T, U_pred, cmap='viridis')
ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u(x,t)')
ax.set_title('PINN Solution of 1D Burgers Equation')
plt.savefig('burgers_pinn.png')
plt.show()

# 不同时刻剖面线图
plt.figure(figsize=(8, 4))
for ti in [0, 0.25, 0.5, 0.75, 1.0]:
    idx = np.argmin(np.abs(t - ti))
    plt.plot(x, U_pred[idx], label=f't={ti}')
plt.xlabel('x')
plt.ylabel('u')
plt.legend()
plt.title('Solution Profiles at Different Times')
plt.savefig('burgers_profiles.png')
plt.show()