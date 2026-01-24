import pandas as pd
import matplotlib.pyplot as plt

class Helper:
  def __init__(self, filename) :
    self.filename = filename
    self.df = pd.read_csv(filename)
    
  def split(self, train_ratio):
    df = self.df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    # Split ratio
    train_size = int(len(df) * train_ratio)
    
    # Split
    train_df = df.iloc[:train_size]
    test_df  = df.iloc[train_size:]
    
    # Save files
    train_df.to_csv("../data/train1.csv", index=False)
    test_df.to_csv("../data/train2.csv", index=False)
    
  def minmax(self):
    # Compute min and max
    battery_min = self.df["battery_power"].min()
    battery_max = self.df["battery_power"].max()
    
    ram_min = self.df["ram"].min()
    ram_max = self.df["ram"].max()

    print("Battery power min:", battery_min)
    print("Battery power max:", battery_max)
    print("RAM min:", ram_min)
    print("RAM max:", ram_max)
    
  def clean(self, names):
    # Keep only required columns
    df = self.df[names]

    # Save cleaned CSV
    df.to_csv("test.csv", index=False)
    
  def plot(self, isoutput):
    df = self.df
    # Separate by price
    cheap = df[self.df["price_range"] == 0]
    expensive = df[self.df["price_range"] == 1]
    
    if isoutput:
      Outcheap = df[self.df["output"] == 0]
      Outexpensive = df[self.df["output"] == 1]
    
    
    # Plot
    plt.scatter(cheap["battery_power"], cheap["ram"], color="red", label="cheap", s=100)
    plt.scatter(expensive["battery_power"], expensive["ram"], color="black", label="Expensive", s=100)
    
    if isoutput:
      plt.scatter(Outcheap["battery_power"], Outcheap["ram"], color="red", label="Outputcheap", s=100)
      plt.scatter(Outexpensive["battery_power"], Outexpensive["ram"], color="black", label="OutputExpensive", s=100)
    
    # Labels and title
    plt.xlabel("battery_power")
    plt.ylabel("ram")
    plt.title("battery_power vs ram by price")
    plt.legend()
    
    # Save plot as PNG
    plt.savefig("plot.png", dpi=150)
    print("done")
    