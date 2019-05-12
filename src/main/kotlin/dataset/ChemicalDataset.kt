package dataset

class ChemicalDataset : CustomDataset(Scaler.SCALE_SEPARATELY, "datasets/chemical/chemical.txt", 0.05)