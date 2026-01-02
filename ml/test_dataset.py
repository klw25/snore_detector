from dataset import SnoreDataset

ds = SnoreDataset()

print("Total samples:", len(ds))

mel, label = ds[0]

print("Mel shape:", mel.shape)
print("Label:", label)
