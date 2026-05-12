# ============================================
# Student Grade Analyzer with AI Prediction
# Created by: Ankit (@Ankrypt)
# GitHub: github.com/ankrypt
# ============================================

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings("ignore")

# ── 1. Load Data ──────────────────────────────
print("=" * 45)
print("   Student Grade Analyzer by Ankrypt")
print("=" * 45)

df = pd.read_csv("students.csv")
print("\n📋 Loaded Data:")
print(df.to_string(index=False))

# ── 2. Basic Stats ────────────────────────────
print("\n📊 Basic Statistics:")
print(df[["Math", "Science", "English"]].describe().round(2))

# ── 3. Visualisation — Bar Chart ──────────────
avg_marks = df[["Math", "Science", "English"]].mean()

plt.figure(figsize=(8, 5))
bars = plt.bar(avg_marks.index, avg_marks.values,
               color=["#4CAF50", "#2196F3", "#FF9800"], width=0.5)
plt.title("📊 Average Marks by Subject", fontsize=14, fontweight="bold")
plt.ylabel("Average Marks")
plt.ylim(0, 100)
for bar, val in zip(bars, avg_marks.values):
    plt.text(bar.get_x() + bar.get_width()/2,
             bar.get_height() + 1,
             f"{val:.1f}", ha="center", fontweight="bold")
plt.tight_layout()
plt.savefig("avg_marks_chart.png", dpi=150)
plt.show()
print("\n✅ Chart saved as avg_marks_chart.png")

# ── 4. Visualisation — Pass vs Fail Pie ───────
result_counts = df["Result"].value_counts()

plt.figure(figsize=(6, 6))
plt.pie(result_counts,
        labels=result_counts.index,
        autopct="%1.1f%%",
        colors=["#4CAF50", "#f44336"],
        startangle=90,
        textprops={"fontsize": 13})
plt.title("🎯 Pass vs Fail Distribution",
          fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("pass_fail_chart.png", dpi=150)
plt.show()
print("✅ Chart saved as pass_fail_chart.png")

# ── 5. Train AI Model ─────────────────────────
print("\n🤖 Training AI Model...")

X = df[["Math", "Science", "English"]]
y = (df["Result"] == "Pass").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred) * 100

print(f"\n✅ Model Accuracy: {acc:.1f}%")
print("\n📋 Classification Report:")
print(classification_report(y_test, y_pred,
      target_names=["Fail", "Pass"]))

# ── 6. Predict a New Student ──────────────────
print("\n" + "=" * 45)
print("   🔮 Predict a New Student's Result")
print("=" * 45)

try:
    math    = int(input("\nEnter Math marks (0-100):    "))
    science = int(input("Enter Science marks (0-100): "))
    english = int(input("Enter English marks (0-100): "))

    new_student = [[math, science, english]]
    prediction  = model.predict(new_student)[0]
    probability = model.predict_proba(new_student)[0]

    avg = (math + science + english) / 3

    print("\n" + "─" * 45)
    print(f"📌 Average Score : {avg:.1f}%")
    print(f"✅ Pass Chance   : {probability[1]*100:.1f}%")
    print(f"❌ Fail Chance   : {probability[0]*100:.1f}%")
    print(f"\n🎯 Prediction    : {'🎉 PASS' if prediction == 1 else '😔 FAIL'}")
    print("─" * 45)

except ValueError:
    print("⚠️  Please enter valid numbers!")

print("\n✅ Analysis complete! Check the charts saved in your folder.")
print("📺 Follow Ankrypt on YouTube for more projects!")
