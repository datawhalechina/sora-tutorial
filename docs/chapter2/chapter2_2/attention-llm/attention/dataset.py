class Dataset:

    def __init__(self):
        self.data = []

    def build(self, df, fea_col: str, label_col: str):
        for i in range(len(df)):
            v = df.iloc[i]
            text = v[fea_col]
            label = v[label_col]
            item = {
                "text": text,
                "label": label
            }
            self.data.append(item)
    
    def __iter__(self):
        for v in self.data:
            yield v

    def __getitem__(self, index: int):
        return self.data[index]

    def __len__(self):
        return len(self.data)