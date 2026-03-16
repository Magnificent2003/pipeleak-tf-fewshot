import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import FuncFormatter, MultipleLocator

# 1. 读入并整理数据
df = pd.read_excel("figure/figure5.2.1.xlsx")
clean = df.rename(columns={
    'Unnamed: 1': 'Metric',
    'Unnamed: 2': 'Model',
    'Unnamed: 3': 'Score'
})[['Metric', 'Model', 'Score']]
clean['Metric'] = clean['Metric'].ffill()
clean = clean.dropna(subset=['Model']).reset_index(drop=True)

plt.rcParams['font.family'] = 'Times New Roman'

metrics = ['F1', 'Recall']
subsets = [clean[clean['Metric'] == m] for m in metrics]

n_per_group = [len(s) for s in subsets]
width = 0.8
x_positions = []
current_x = 0
for n in n_per_group:
    xs = list(range(current_x, current_x + n))
    x_positions.extend(xs)
    current_x += n + 1

scores = []
models = []
for s in subsets:
    scores.extend(s['Score'].tolist())
    models.extend(s['Model'].tolist())

fig, ax = plt.subplots(figsize=(6, 3))

model_list = list(dict.fromkeys(models))
base_colors = ['#4C72B0', '#DD8452', '#999999',
               '#F0C808', '#55A868', '#8172B2']
palette = {m: base_colors[i % len(base_colors)] for i, m in enumerate(model_list)}
bar_colors = [palette[m] for m in models]

bars = ax.bar(x_positions, scores,
              width=width,
              color=bar_colors,
              edgecolor='none')

ax.set_ylim(0.75, 0.95)
ax.set_ylabel('Score (%)', fontweight='bold')

ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y*100:.0f}"))
ax.yaxis.set_major_locator(MultipleLocator(0.05))

ax.yaxis.grid(True, linestyle='--', linewidth=0.5, alpha=0.4)
ax.set_axisbelow(True)

centers = []
start = 0
for n in n_per_group:
    centers.append(start + (n - 1) / 2)
    start += n + 1

ax.set_xticks(centers)
ax.set_xticklabels(['Binary-F1', 'Binary-Recall'], fontweight='bold')

for spine in ax.spines.values():
    spine.set_visible(True)

ax.set_title('')
ax.set_xlabel('')

# 顶部百分比标签
for bar in bars:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2,
            h + 0.002,
            f"{h*100:.1f}",
            ha='center', va='bottom',
            fontsize=9)

handles = [mpatches.Patch(color=palette[m], label=m) for m in model_list]
ax.legend(handles=handles,
          frameon=False,
          fontsize=9,
          ncol=3,
          loc='upper center',
          bbox_to_anchor=(0.5, -0.10))

plt.tight_layout()
plt.savefig("figure/figure5.2.1_python.png", bbox_inches='tight', dpi=600, transparent=True)
plt.savefig("figure/figure5.2.1_python.svg", bbox_inches='tight')
plt.show()
