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

vec = run["vectorize"].task.data.vec
train_dataset = run["describe"].task.data.train_dataset
train_dataset

