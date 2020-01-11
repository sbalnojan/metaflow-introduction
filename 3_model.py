import gluonnlp as nlp
from metaflow import FlowSpec,step, Parameter
import pandas as pd

class DataFlow(FlowSpec):
    max_features = Parameter('max_features',
                      help='Max no. of features to use',
                      default=10000)

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

        train_dataset, test_dataset = [
            nlp.data.IMDB(root=target_dir, segment=segment) for segment in ("train", "test")
        ]
        self.train_dataset = pd.DataFrame(train_dataset._data, columns=["text", "score"])
        self.test_dataset = pd.DataFrame(test_dataset._data, columns=["text", "score"])

        self.next(self.describe)

    @step
    def describe(self):
        df = self.train_dataset
        print(df.head(5))
        print(df.describe())
        self.next(self.vectorize)

    @step
    def vectorize(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        max_features = self.max_features
        self.vec = TfidfVectorizer(max_features=max_features)
        self.vec.fit(self.train_dataset["text"])
        self.next(self.fit_train, self.fit_test)

    @step
    def fit_train(self):
        transformed_data = self.vec.transform(self.train_dataset["text"])
        self.transformed_data = pd.DataFrame(transformed_data.toarray())

        self.next(self.join)

    @step
    def fit_test(self):
        transformed_data = self.vec.transform(self.test_dataset["text"])
        self.transformed_data = pd.DataFrame(transformed_data.toarray())
        self.next(self.join)

    @step
    def join(self, inputs):
        print('a is %s' % inputs.fit_train.transformed_data.head())
        print('b is %s' % inputs.fit_test.transformed_data.head())
        self.next(self.end)

    @step
    def end(self):
        print("All done")

if __name__ == "__main__":
    DataFlow()