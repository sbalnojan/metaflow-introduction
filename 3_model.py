import gluonnlp as nlp
from metaflow import FlowSpec,step, Parameter, current
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
        self.y = self.train_dataset["score"]
        self.next(self.fit_clf)

    @step
    def fit_test(self):
        transformed_data = self.vec.transform(self.test_dataset["text"])
        self.transformed_data = pd.DataFrame(transformed_data.toarray())
        self.y = self.test_dataset["score"]
        self.next(self.score)

    @step
    def fit_clf(self):
        print('a is %s' % self.transformed_data.head())
        print("Now let's train....\n")

        from sklearn.ensemble import RandomForestClassifier
        clf = RandomForestClassifier(max_depth=5, random_state=0)
        clf.fit(self.transformed_data, self.y)
        self.clf = clf
        self.next(self.score)

    @step
    def score(self, inputs):
        predictions = inputs.fit_clf.clf.predict(inputs.fit_test.transformed_data)

        from sklearn.metrics import f1_score

        f1_score = f1_score(inputs.fit_test.y, predictions, average="micro")
        print(f1_score)
        self.clf = inputs.fit_clf.clf
        self.next(self.end)

    @step
    def end(self):
        print("All done, let us version the important artifacts")
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        version_id = current.flow_name + current.run_id
        model = self.clf
        parameters = self.max_features
        print(f"we can version it as {version_id}")


if __name__ == "__main__":
    DataFlow()