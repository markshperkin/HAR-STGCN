import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('trainresults_10c.csv')

# acc vs Epoch
plt.figure(figsize=(8, 5))
df['Train Acc'] = df['Train Acc'] * 100
df['Val Acc']   = df['Val Acc']   * 100

plt.plot(df['Epoch'], df['Train Acc'], label='Train Accuracy', color='yellow')
plt.plot(df['Epoch'], df['Val Acc'],   label='Validation Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('accuracy_plot.png')
plt.show()

# loss vs epoch
plt.figure(figsize=(8, 5))
plt.plot(df['Epoch'], df['Train Loss'], label='Train Loss', color='yellow')
plt.plot(df['Epoch'], df['Val Loss'],   label='Validation Loss', color='green')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('loss_plot.png')
plt.show()
