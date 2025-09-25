import matplotlib.pyplot as plt
import numpy as np

# -------------------------
# Data
# -------------------------
projects = ['fastapi', 'flask', 'keras', 'requests', 'scrapy']
commits = [934, 1967, 9256, 2749, 6132]

partitions = ['Train', 'Test']
no_debt = [13450, 5909]
debt = [1274, 405]

# Colorblind-friendly palette
colors_pie = ['#0072B2', '#E69F00', '#56B4E9', '#009E73', '#D55E00']
colors_bar = ['#0072B2', '#D55E00']

# -------------------------
# Figure setup
# -------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14,6))

# Pie chart — Commit Distribution
def autopct_with_counts(pct, allvals):
    absolute = int(round(pct/100.*np.sum(allvals)))
    return f"{pct:.1f}%\n({absolute})"

ax1.pie(
    commits, 
    labels=projects, 
    autopct=lambda pct: autopct_with_counts(pct, commits), 
    startangle=140, 
    colors=colors_pie,
    wedgeprops={'edgecolor':'white', 'linewidth':1.2},
    textprops={'fontsize':12, 'weight':'bold', 'color':'black'}
)
ax1.set_title('Commit Distribution Across Projects', fontsize=14, weight='bold')

# Grouped bar chart — Debt vs No-Debt
x = np.arange(len(partitions))
width = 0.35

ax2.bar(x - width/2, no_debt, width, label='No Debt', color=colors_bar[0], edgecolor='white', linewidth=1.2)
ax2.bar(x + width/2, debt, width, label='Debt', color=colors_bar[1], edgecolor='white', linewidth=1.2)

ax2.set_ylabel('Number of Commits', fontsize=12, weight='bold')
ax2.set_title('Debt vs No-Debt Commits', fontsize=14, weight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(partitions, fontsize=12, weight='bold')
ax2.legend(fontsize=12)
ax2.grid(axis='y', linestyle='--', alpha=0.5)

plt.tight_layout()

# -------------------------
# Save as high-resolution PNG and PDF
# -------------------------
png_path = './data_split_slide.png'
pdf_path = './data_split_slide.pdf'
plt.savefig(png_path, dpi=300, bbox_inches='tight')
plt.savefig(pdf_path, dpi=300, bbox_inches='tight')

png_path, pdf_path
plt.show()