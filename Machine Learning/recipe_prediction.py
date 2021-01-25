import pandas as pd
import numpy as np
import pickle
import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

LDA_FILE = 'lda.pickle'
RF_FILE = 'rf.pickle'

RAW_recipes = pd.read_csv('RAW_recipes.csv')
print(len(RAW_recipes))

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
print(verbs.most_common()[:100])

tokens = wordcounts.most_common()[:5000]
token_list = [tok[0] for tok in tokens]
token_set = set(token_list)
token2idx = {tokens[i][0]:i for i in range(len(tokens))}


recipes = []
keywords = []

for i in range(len(RAW_recipes)):
    words = RAW_recipes.loc[i, 'steps'].replace('[', ' [ ').replace(']', ' ] ').replace("'", " ' ").replace(':', ' : ').split()
    for w in set(words):
        if w in token_set:
            recipes.append(i)
            keywords.append(token2idx[w])
print(len(recipes))

from scipy.sparse import csr_matrix
data = [1 for _ in recipes]
X = csr_matrix((data, (recipes, keywords)), shape=(len(RAW_recipes), 5000))

metric = []

from sklearn.metrics import silhouette_score
from sklearn.decomposition import LatentDirichletAllocation

for i in range(10):
    lda = LatentDirichletAllocation(n_components= i + 1, random_state=0, verbose = 2)
    lda.fit(X)
    transformed = lda.transform(X)
    perplexity = lda.bound_
    print(perplexity)
    metric.append(perplexity)

plt.bar(np.arange(10) + 1, np.array(metric))
plt.show()
'''
with open(LDA_FILE, 'wb') as file:
    pickle.dump(transformed, file)

    
for i in range(4):
    importances = transformed[:, i]
    indices = np.argsort(importances)[-50:]
    words = ' '.join([RAW_recipes.loc[i, 'name'] for i in indices])
    wordcloud = WordCloud().generate(words)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
'''

'''
y = np.zeros((len(RAW_recipes), 3))
y[RAW_recipes.loc[:, 'minutes'] < 30, 0] = 1
y[(RAW_recipes.loc[:, 'minutes'] >= 30) & (RAW_recipes.loc[:, 'minutes'] < 120), 1] = 1
y[RAW_recipes.loc[:, 'minutes'] >= 120, 2] = 1
print(y.shape)

from sklearn.ensemble import RandomForestClassifier
regr = RandomForestClassifier(n_estimators = 100, max_depth=20, random_state=0, oob_score=True, verbose = 2, n_jobs = -1)
regr.fit(X, y)

with open(RF_FILE, 'wb') as file:
    pickle.dump(regr, file)
    



ind = np.argpartition(regr.oob_decision_function_[0][:, 1], -20)[-20:]
print([RAW_recipes.loc[i, 'name'] for i in ind])
print(regr.oob_decision_function_[0][ind, :])
ind = np.argpartition(regr.oob_decision_function_[1][:, 1], -20)[-20:]
print('\n\n', [RAW_recipes.loc[i, 'name'] for i in ind])
print(regr.oob_decision_function_[1][ind, :])
ind = np.argpartition(regr.oob_decision_function_[2][:, 1], -20)[-20:]
print('\n\n', [RAW_recipes.loc[i, 'name'] for i in ind])
print(regr.oob_decision_function_[2][ind, :])
print(regr.oob_score_)

importances = regr.feature_importances_
indices = np.argsort(importances)[-20:]

plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [token_list[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()
'''
