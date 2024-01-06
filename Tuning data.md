from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

# Membuat pipeline dengan CountVectorizer dan MultinomialNB
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Parameter yang ingin diuji
param_grid = {'countvectorizer__ngram_range': [(1, 1), (1, 2), (2, 2)],
              'multinomialnb__alpha': [0.1, 1.0, 2.0]}

# Membuat objek GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5)

# Melatih model dengan Grid Search
grid.fit(X_train, y_train)

# Melihat parameter terbaik
print("Parameter terbaik:", grid.best_params_)

# Melihat skor terbaik
print("Skor terbaik:", grid.best_score_)
