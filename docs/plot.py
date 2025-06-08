import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("time.csv")

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(df["cores"], df["speedup"], marker="o")
plt.title("Parallel Speedup vs. Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Speedup")
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(df["cores"], df["efficiency"], marker="o", color="orange")
plt.title("Parallel Efficiency vs. Threads")
plt.xlabel("Number of Threads")
plt.ylabel("Efficiency")
plt.grid(True)

plt.tight_layout()
plt.savefig("plot.png")
