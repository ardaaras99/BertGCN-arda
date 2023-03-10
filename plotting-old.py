#%%""
# EXTRA PLOT SECTION
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from matplotlib import rcParams, rc

# rc("text", usetex=True)
# # rc("axes", linewidth=2)
# rc("font", weight="bold")
plt.rcParams["text.usetex"] = True
plt.rcParams["axes.labelweight"] = "bold"
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

x = [
    1 * pow(10, 0),
    1 * pow(10, 0),
    2 * pow(10, 0),
    3 * pow(10, 0),
    7 * pow(10, 1),
    5 * pow(10, 2),
]
y = [95.7, 96.5, 98.3, 96.96, 97.1, 98.2]
s = [
    "$\mathbf{Type \, I}$",
    "$\mathbf{Type \, II}$",
    "$\mathbf{Type \, III}$",
    "$\mathbf{HeteGCN}$",  # hetegcn
    "$\mathbf{TextGCN}$",  # textgcn
    "$\mathbf{RoBERTaGCN}$",  # bertgcn
]
markers = ["d", "v", "s", "*", "^", "d", "s"]


fig, ax = plt.subplots(figsize=(10, 5))

for xp, yp, m in zip(x, y, markers):
    ax.scatter(xp, yp, marker=m, s=150)  # type: ignore
    ax.set_xscale("log")

ax.set_xticks([10**0, 10**1, 10**2, 10**3, 10**4])

ax.set_xticklabels(
    [
        "$\mathbf{10^{0}}$",
        "$\mathbf{10^{1}}$",
        "$\mathbf{10^{2}}$",
        "$\mathbf{10^{3}}$",
        "$\mathbf{10^{4}}$",
    ],
    rotation=0,
)

ax.set_yticks([95, 96, 97, 98, 99, 100])
ax.set_yticklabels(
    [
        "$\mathbf{95}$",
        "$\mathbf{96}$",
        "$\mathbf{97}$",
        "$\mathbf{98}$",
        "$\mathbf{99}$",
        "$\mathbf{100}$",
    ],
    rotation=0,
)


ax.set_xlim([10**-0.5, 10**4])
ax.set_ylim([95.0, 100.0])

fontsize_m = 16
# No Name
# ax.annotate(s[0], (x[0] - 0.3, y[0] - 0.4), fontsize=fontsize_m)
# ax.annotate(s[1], (x[1] - 0.3, y[1] - 0.4), fontsize=fontsize_m)
# ax.annotate(s[2], (x[2] - 0.8, y[2] - 0.4), fontsize=fontsize_m)

# ax.annotate(s[3], (x[3] - 100, y[3] - 0.4), fontsize=fontsize_m)
# ax.annotate(s[4], (x[4] - 0.4, y[4] - 0.4), fontsize=16)
# ax.annotate(s[5], (x[5] - 8, y[5] - 0.5), fontsize=16)
# ax.annotate(s[6], (x[6] - 50, y[6] - 0.4), fontsize=16)

# With Name
ax.annotate(s[0], (x[0] - 0.3, y[0] - 0.4), fontsize=fontsize_m)
ax.annotate(s[1], (x[1] - 0.3, y[1] - 0.4), fontsize=fontsize_m)
ax.annotate(s[2], (x[2] - 0.8, y[2] + 0.3), fontsize=fontsize_m)

ax.annotate(s[3], (x[3] - 1.4, y[3] - 0.4), fontsize=16)
ax.annotate(s[4], (x[4] - 31, y[4] - 0.5), fontsize=16)
ax.annotate(s[5], (x[5] - 100, y[5] - 0.5), fontsize=12)

fontsize = 22
for tick in ax.xaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight("bold")
for tick in ax.yaxis.get_major_ticks():
    tick.label1.set_fontsize(fontsize)
    tick.label1.set_fontweight("bold")

font = {
    "color": "black",
    "weight": "bold",
    "size": 22,
}

ax.set_ylabel("$\mathbf{Test \, Accuracy \, (\%)}$", fontdict=font)
ax.set_xlabel("$\mathbf{Relative \, Training \, Time}$", fontdict=font)
plt.grid(True)

plt.show()
# fig.savefig("results/relative_train_time.eps", format="eps", bbox_inches="tight")
# fig.savefig("results/relative_train_time.png", format="png", bbox_inches="tight")


# annotate altına yap
# test accuracy rel train time küçük, büyük olması lazım
# rbgcn taşyor
# ms ekle

#%%
