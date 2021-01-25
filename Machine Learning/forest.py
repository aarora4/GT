import pandas as pd
import numpy as np

RAW_recipes = pd.read_csv('RAW_recipes.csv')
RAW_interactions = pd.read_csv('RAW_interactions.csv')

idx2recipe = [RAW_recipes.loc[i, 'id'] for i in range(len(RAW_recipes))]
recipe2idx = {idx2recipe[i]:i for i in range(len(idx2recipe))}
idx2user = list(set([RAW_interactions.loc[i, 'user_id'] for i in range(len(RAW_interactions))]))
user2idx = {idx2user[i]:i for i in range(len(idx2user))}

print(len(idx2recipe))

ingred2idx = {}
idx2ingred = []
recipes = []
ingreds = []
for i in range(len(RAW_recipes)):
    s = RAW_recipes.loc[i, 'ingredients'].replace('[', '').replace(']', '').replace(',', '').replace("'", '').split(" ")
    for ingred in s:
        if len(ingred) > 0 and not ingred.isspace(): 
            if ingred not in ingred2idx:
                ingred2idx[ingred] = len(ingred2idx)
                idx2ingred.append(ingred)
            recipes.append(recipe2idx[RAW_interactions.loc[i, 'recipe_id']])
            ingreds.append(ingred2idx[ingred])
print(len(ingreds))

data = [1 for _ in range(len(recipes))]

from scipy.sparse import csr_matrix
matrix = csr_matrix((data, (recipes, ingreds)), shape=(len(recipe2idx), len(ingred2idx)))

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components=100, n_iter=200, random_state=42)
svd.fit(matrix)
X = svd.transform(matrix)
print(X.shape)

y = np.zeros((len(recipe2idx)))
c = np.zeros((len(recipe2idx)))
for i in range(len(RAW_interactions)):
    y[recipe2idx[RAW_interactions.loc[i, 'recipe_id']]] += RAW_interactions.loc[i, 'rating']
    c[recipe2idx[RAW_interactions.loc[i, 'recipe_id']]] += 1
y = y/c
print(y.shape)

from sklearn.ensemble import RandomForestRegressor
regr = RandomForestRegressor(n_estimators = 100, max_depth=20, random_state=0, oob_score=True, verbose = 2, n_jobs = -1)
regr.fit(X, y)
print(regr.oob_score_)