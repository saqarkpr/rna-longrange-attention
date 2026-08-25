import matplotlib.pyplot as plt

bins = ["20-60", "60-150", "150-300", "300-500"]
baseline = [0.3975, 0.3598, 0.2575, 0.2115]
kernel = [0.4353, 0.3817, 0.3540, 0.3136]

fig, ax = plt.subplots(figsize=(7, 5))
ax.plot(bins, baseline, marker='o', linewidth=2.5, markersize=8, label='Standard Transformer (baseline)', color='#888888')
ax.plot(bins, kernel, marker='o', linewidth=2.5, markersize=8, label='Kernel-Biased Attention', color='#1F3B57')

ax.set_xlabel('Sequence length bin', fontsize=12)
ax.set_ylabel('Pair-match accuracy', fontsize=12)
ax.set_title('Long-Range Pairing Recovery vs. Sequence Length', fontsize=13, fontweight='bold')
ax.set_ylim(0, 0.5)
ax.legend(fontsize=11, loc='upper right')
ax.grid(alpha=0.3)

for i, (b, k) in enumerate(zip(baseline, kernel)):
    ax.annotate(f'+{(k-b)*100:.1f}pp', xy=(i, k+0.02), fontsize=9, ha='center', color='#1F3B57', fontweight='bold')

plt.tight_layout()
plt.savefig('pair_match_accuracy_by_length.png', dpi=150)
print("saved plot")
