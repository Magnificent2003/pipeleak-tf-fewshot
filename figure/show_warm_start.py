import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ===== 全局字体设置 =====
mpl.rcParams['font.family'] = 'Times New Roman'
mpl.rcParams['mathtext.fontset'] = 'stix'   # 数学公式更接近 Times
mpl.rcParams['axes.unicode_minus'] = False

# ===== Epoch axis =====
epochs = np.linspace(0, 200, 1000)

# ===== Hyper-parameters (illustrative) =====
T_warm_b = 50        # small warm-up
T_warm_cons = 100     # larger warm-up

lambda_b = 0.6       # small weight
lambda_cons = 1.0    # large weight

# ===== Schedules =====
lambda_b_t = np.where(epochs < T_warm_b, 0.0, lambda_b)
lambda_cons_t = np.where(epochs < T_warm_cons, 0.0, lambda_cons)

# ===== Plot =====
plt.figure(figsize=(4.5, 3))

plt.step(
    epochs, lambda_b_t,
    where="post",
    linewidth=2,
    label=r'$\lambda_b(t)$'
)

plt.step(
    epochs, lambda_cons_t,
    where="post",
    linewidth=2,
    linestyle='--',
    label=r'$\lambda_{\mathrm{cons}}(t)$'
)

# ===== Annotations =====
plt.axvline(T_warm_b, linestyle=':', linewidth=1)
plt.axvline(T_warm_cons, linestyle=':', linewidth=1)

plt.text(T_warm_b + 1, lambda_b + 0.03, r'$T_{\mathrm{warm}}^{b}$', fontsize=9)
plt.text(T_warm_cons + 1, lambda_cons + 0.03, r'$T_{\mathrm{warm}}^{cons}$', fontsize=9)

# ===== Styling =====
plt.xlabel('Epoch')
plt.ylabel('Loss weight')
plt.ylim(-0.05, 1.15)
plt.xlim(0, 200)

plt.legend(frameon=False)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()
