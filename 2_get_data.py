import gluonnlp as nlp
from metaflow import FlowSpec,step
import pandas as pd

class DataFlow(FlowSpec):
    @step
    def start(self):
        print("Beginning our example")
        self.next(self.hello)

    @step

    def hello(self):
        """
        Step will download the imdb bin. classification dataset
        """
        target_dir = "./data/imdb"
        print(f"Will now download the data to folder {target_dir}")

        self.train_dataset, self.test_dataset = [
            nlp.data.IMDB(root=target_dir, segment=segment) for segment in ("train", "test")
        ]

        self.next(self.describe)

    @step
    def describe(self):
        df = pd.DataFrame(self.train_dataset._data, columns=["text", "score"])
        print(df.head(5))
        print(df.describe())
        self.next(self.end)

    @step
    def end(self):
        print("All done")

if __name__ == "__main__":
    DataFlow()