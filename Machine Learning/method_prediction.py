import pandas as pd
import numpy as np
import pickle
import tqdm

LDA_FILE = 'lda.pickle'
RF_FILE = 'ingred2step.pickle'
KEYWORDS = 1000

RAW_recipes = pd.read_csv('RAW_recipes.csv')
print(len(RAW_recipes))

idx2recipe = [RAW_recipes.loc[i, 'id'] for i in range(len(RAW_recipes))]
recipe2idx = {idx2recipe[i]:i for i in range(len(idx2recipe))}

ingred2idx = {}
idx2ingred = []
x_recipes = []
ingreds = []
for i in range(len(RAW_recipes)):
    s = RAW_recipes.loc[i, 'ingredients'].replace('[', '').replace(']', '').replace(',', '').replace("'", '').split(" ")
    for ingred in s:
        if len(ingred) > 0 and not ingred.isspace(): 
            if ingred not in ingred2idx:
                ingred2idx[ingred] = len(ingred2idx)
                idx2ingred.append(ingred)
            x_recipes.append(i)
            ingreds.append(ingred2idx[ingred])
print(len(idx2ingred))

data = [1 for _ in x_recipes]
from scipy.sparse import csr_matrix
X = csr_matrix((data, (x_recipes, ingreds)), shape=(len(recipe2idx), len(ingred2idx)))

from collections import Counter
import nltk
from nltk.corpus import stopwords
wordcounts = Counter()
verbs = Counter()
stop = set(stopwords.words('english'))
symb = [',', '[', ']', '/', '\\', '(', ')', "'", ':']
stop.update(symb)
calories = []
for i in tqdm.tqdm(range(len(RAW_recipes))):
    words = RAW_recipes.loc[i, 'steps'].replace('[', ' ').replace(']', ' ').replace("'", " ").replace(':', ' ').split()
    nut = RAW_recipes.loc[i, 'nutrition'].replace('[', ' ').replace(']', ' ').replace("'", " ").replace(':', ' ').split(',')
    calories.append(float(nut[0]))
    for w in words:
        if w not in stop and not w.isnumeric():
            wordcounts[w] += 1

tokens = wordcounts.most_common()[:KEYWORDS]
token_set = set([tok[0] for tok in tokens])
token2idx = {tokens[i][0]:i for i in range(len(tokens))}


y_recipes = []
keywords = []

Y = np.zeros((len(RAW_recipes), KEYWORDS))
for i in range(len(RAW_recipes)):
    words = RAW_recipes.loc[i, 'steps'].replace('[', ' [ ').replace(']', ' ] ').replace("'", " ' ").replace(':', ' : ').split()
    for w in set(words):
        if w in token_set:
            Y[i, token2idx[w]] = 1


from sklearn.ensemble import RandomForestClassifier
regr = RandomForestClassifier(n_estimators = 100, max_depth=20, random_state=0, oob_score=True, verbose = 2, n_jobs = -1)
regr.fit(X, Y)

with open(RF_FILE, 'wb') as file:
    pickle.dump(regr, file)
    
print(regr.oob_score_)