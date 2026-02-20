import pandas as pd

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    print("Train shape:", train.shape)
    print("Test shape:", test.shape)
    print("\nTrain columns:\n", train.columns.tolist())
    print("\nMissing values:\n", train.isnull().sum().sort_values(ascending=False).head(10))
    return train, test

if __name__ == "__main__":
    load_data()
