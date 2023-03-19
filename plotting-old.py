#%%""
# EXTRA PLOT SECTION
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib

from matplotlib import rcParams, rc

plt.rcParams["text.usetex"] = True
plt.rcParams["axes.labelweight"] = "bold"
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = [r"\usepackage{amsmath}"]

# Obtained Results
r8_x = [0.673, 0.552, 0.833, 0.640, 3.21, 105.63]
r8_y = [92.14, 92.76, 98.72, 94.39, 97.1, 98.2]

r52_x = [0.185, 0.149, 0.158, 0.160, 0.822, 152.07]
r52_y = [89.74, 87.21, 96.25, 91.93, 93.6, 96.1]

ohsumed_x = [0.162, 0.152, 0.152, 0.155, 0.87, 109.34]
ohsumed_y = [58.08, 60.30, 71.70, 62.46, 68.4, 72.8]

mr_x = [0.158, 0.138, 0.121, 0.121, 0.72, 218.32]
mr_y = [74.06, 77.83, 89.44, 74.23, 76.7, 89.7]

s = [
    "$\mathbf{Type \, I}$",
    "$\mathbf{Type \, II}$",
    "$\mathbf{Type \, III}$",
    "$\mathbf{HeteGCN}$",  # hetegcn
    "$\mathbf{TextGCN}$",  # textgcn
    "$\mathbf{RoBERTaGCN}$",  # bertgcn
]
markers = ["d", "v", "s", "*", "^", "d", "s"]

# %%
# FOR MR

fig, ax = plt.subplots(figsize=(10, 5))
x, y = mr_x, mr_y
for xp, yp, m in zip(x, y, markers):
    ax.scatter(xp, yp, marker=m, s=150)  # type: ignore
    ax.set_xscale("log")

ax.set_xticks([10 ** (-1), 10**0, 10**1, 10**2, 10**3])
ax.set_xticklabels(
    [
        "$\mathbf{10^{-1}}$",
        "$\mathbf{10^{0}}$",
        "$\mathbf{10^{1}}$",
        "$\mathbf{10^{2}}$",
        "$\mathbf{10^{3}}$",
    ],
    rotation=0,
)
y_ticks_arr = [72, 76, 80, 84, 88, 92]
ax.set_yticks(y_ticks_arr)
ax.set_yticklabels(
    [
        "$\mathbf{72}$",
        "$\mathbf{76}$",
        "$\mathbf{80}$",
        "$\mathbf{84}$",
        "$\mathbf{88}$",
        "$\mathbf{92}$",
    ],
    rotation=0,
)
ax.set_xlim([10**-1.8, 10**3])
ax.set_ylim([70.0, 92.0])

fontsize_m = 16
# With Name
ax.annotate(s[0], (x[0] + 0.002, y[0] - 1.7), fontsize=fontsize_m)  # type1
ax.annotate(s[1], (x[1] - 0.05, y[1] + 1.2), fontsize=fontsize_m)  # type2
ax.annotate(s[2], (x[2] - 0.05, y[2] - 2), fontsize=fontsize_m)  # type3

ax.annotate(s[3], (x[3] - 0.09, y[3] + 0.7), fontsize=16)  # hetegcn
ax.annotate(s[4], (x[4] - 0.3, y[4] + 0.9), fontsize=16)  # textgcn
ax.annotate(s[5], (x[5] - 120, y[5] - 2), fontsize=12)  # robertagcn

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
fig.savefig("results/mr_relative-train-time.eps", format="eps", bbox_inches="tight")

#%%

# For Ohsumed
fig, ax = plt.subplots(figsize=(10, 5))
x, y = ohsumed_x, ohsumed_y
for xp, yp, m in zip(x, y, markers):
    ax.scatter(xp, yp, marker=m, s=150)  # type: ignore
    ax.set_xscale("log")

ax.set_xticks([10 ** (-1), 10**0, 10**1, 10**2, 10**3])
ax.set_xticklabels(
    [
        "$\mathbf{10^{-1}}$",
        "$\mathbf{10^{0}}$",
        "$\mathbf{10^{1}}$",
        "$\mathbf{10^{2}}$",
        "$\mathbf{10^{3}}$",
    ],
    rotation=0,
)
y_ticks_arr = [56, 60, 64, 68, 72]
ax.set_yticks(y_ticks_arr)
ax.set_yticklabels(
    [
        "$\mathbf{56}$",
        "$\mathbf{60}$",
        "$\mathbf{64}$",
        "$\mathbf{68}$",
        "$\mathbf{72}$",
    ],
    rotation=0,
)
ax.set_xlim([10**-1.8, 10**3])
ax.set_ylim([56.0, 74.0])

fontsize_m = 16
# With Name
ax.annotate(s[0], (x[0] - 0.05, y[0] - 1.7), fontsize=fontsize_m)  # type1
ax.annotate(s[1], (x[1] - 0.05, y[1] - 1.2), fontsize=fontsize_m)  # type2
ax.annotate(s[2], (x[2] - 0.05, y[2] - 2), fontsize=fontsize_m)  # type3

ax.annotate(s[3], (x[3] - 0.09, y[3] + 0.7), fontsize=16)  # hetegcn
ax.annotate(s[4], (x[4] - 0.4, y[4] - 2), fontsize=16)  # textgcn
ax.annotate(s[5], (x[5] - 60, y[5] - 2), fontsize=12)  # robertagcn

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
fig.savefig(
    "results/ohsumed_relative-train-time.eps", format="eps", bbox_inches="tight"
)
