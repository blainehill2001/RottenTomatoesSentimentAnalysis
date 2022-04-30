import pandas as pd
import numpy as np
import warnings
from string import punctuation
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

warnings.simplefilter(action="ignore", category=FutureWarning)
display = pd.options.display
display.max_columns = 1000
display.max_rows = 1000
display.max_colwidth = 199
display.width = 1000

# import and read data
data = pd.read_csv(
    "/Users/blainehill/Desktop/*STOR_565/Final Project/sentiment-analysis-on-movie-reviews/train.tsv",
    sep="\t",
)


"""
longest_review = list(map(current_reviews, lambda x,y: max(len(x), len(y))))
longest_review = initial review
for c in df:


"""
# full_reviews = pd.DataFrame()
temp = []
for i in pd.unique(data["SentenceId"]):
    current_reviews = data[data.SentenceId == i]
    temp.append(current_reviews.iloc[0].to_numpy())
    print(i)
full_reviews = pd.DataFrame(temp, columns=data.columns)
full_reviews = full_reviews.drop(["PhraseId", "SentenceId"], axis=1)


# now we preprocess the reviews


lemma = WordNetLemmatizer()
stop_words = stopwords.words("english")


def text_prep(x):
    # convert to lowercase
    x = str(x).lower()
    # remove weird spacing with regular spacing
    x = re.sub(" +", " ", x).strip()
    # we replace 'n't' with ' not'
    temp = x.split(" ")
    for i in range(len(temp)):
        if temp[i] == "n't":
            temp[i] = "not"
    x = " ".join(temp)
    # we replace ''ll' with 'will'
    temp = x.split(" ")
    for i in range(len(temp)):
        if temp[i] == "'ll":
            temp[i] = "will"
    x = " ".join(temp)
    # we replace '-lrb-' with ''
    temp = x.split(" ")
    for i in range(len(temp)):
        if temp[i] == "-lrb-":
            temp[i] = ""
    x = " ".join(temp)
    # we replace '-rrb-' with ''
    temp = x.split(" ")
    for i in range(len(temp)):
        if temp[i] == "-rrb-":
            temp[i] = ""
    x = " ".join(temp)
    # remove all other non words
    x = re.sub("[^a-zA-Z]+", " ", x).strip()
    tokens = word_tokenize(x)
    words = [t for t in tokens if t not in stop_words]
    # reduce words into their root
    lemmatize = [lemma.lemmatize(w) for w in words]

    return lemmatize


full_reviews["Phrase"] = full_reviews["Phrase"].apply(lambda x: text_prep(x))

# drop any empty rows
full_reviews["Phrase"] = full_reviews["Phrase"].apply(
    lambda x: pd.np.nan if len(x) == 0 else x
)
full_reviews = full_reviews.dropna(subset=["Phrase"])
full_reviews = full_reviews.reset_index(drop=True)

master_counter = Counter("")


def update_master(x):
    temp = Counter(x)
    master_counter.update(temp)


full_reviews["Phrase"].apply(lambda x: update_master(x))
number_of_words_total = sum(master_counter.values())
sorted_words = master_counter.most_common(number_of_words_total)
print(sorted_words)
# full_reviews["Phrase"] = full_reviews["Phrase"].apply(lambda x: x.lower())
# full_reviews["Phrase"] = full_reviews["Phrase"].apply(
#     lambda x: "".join([c for c in x if c not in punctuation])
# )
# full_reviews["Phrase"] = full_reviews["Phrase"].apply(lambda x: re.sub(" +", " ", x))

full_reviews.to_csv(
    "/Users/blainehill/Desktop/*STOR_565/Final Project/sentiment-analysis-on-movie-reviews/train_cleaned.csv"
)
