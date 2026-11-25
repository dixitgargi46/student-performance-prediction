"""Create simple exploratory data analysis graphs."""
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from preprocess import load_data

OUT = Path("outputs/graphs")
OUT.mkdir(parents=True, exist_ok=True)

df, X, y = load_data()
sns.set_theme()

plt.figure(figsize=(7, 4))
sns.countplot(x=y, order=["Needs Improvement", "Average Performance", "Good Performance"])
plt.title("Student Performance Categories")
plt.xticks(rotation=12)
plt.tight_layout()
plt.savefig(OUT / "performance_categories.png")
plt.close()

plt.figure(figsize=(7, 4))
sns.scatterplot(data=df, x="G2", y="G3", hue="performance")
plt.title("Second Period Marks vs Final Marks")
plt.tight_layout()
plt.savefig(OUT / "previous_vs_final_marks.png")
plt.close()

plt.figure(figsize=(8, 5))
sns.heatmap(df[["age", "studytime", "failures", "absences", "G1", "G2", "G3"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(OUT / "correlation_heatmap.png")
plt.close()

print("EDA graphs saved in outputs/graphs/")
