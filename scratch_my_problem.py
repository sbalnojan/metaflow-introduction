from metaflow import Metaflow

flow = list(Metaflow())[0]

run = flow.latest_run

list(run)

train_dataset = run["describe"].task.data.train_dataset
train_dataset

## now let's use the code I used at the first try
## reproduce putitng some error in here...

import pandas as pd
df = pd.DataFrame(train_dataset._data, columns=["text", "score"])
df.head(5)
df.describe()

## -------
## After fixing the error, we can use this/ a notebook to continue prototyping & development,
## Always pushing a finished step into the DAG

from sklearn.feature_extraction.text import TfidfVectorizer
max_features = 500000
vec = TfidfVectorizer(max_features=max_features)
vec.fit(df["text"])



## ---- Lets see the last run
## And retrieve the vectorizer....

flow = list(Metaflow())[0]

run = flow.latest_run

list(run)

y = run["fit_train"].task.data.train_dataset["score"]
df = run["fit_train"].task.data.transformed_data
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(df, y)

X_test = run["fit_test"].task.data.transformed_data
y_test = run["fit_test"].task.data.test_dataset["score"]

print(clf.feature_importances_)

predictions = clf.predict(X_test)

from sklearn.metrics import f1_score

f1_score = f1_score(y_test,predictions, average="micro")