from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Misalkan X adalah data fitur dan y adalah label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat objek model Adaboost dengan Decision Tree sebagai base estimator
base_estimator = DecisionTreeClassifier()
model = AdaBoostClassifier(base_estimator=base_estimator)

# Parameter yang ingin diuji
param_grid = {'n_estimators': [50, 100, 200],
              'learning_rate': [0.01, 0.1, 0.5, 1.0]}

# Membuat objek GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5)

# Melatih model dengan Grid Search
grid.fit(X_train, y_train)

# Melihat parameter terbaik
print("Parameter terbaik:", grid.best_params_)

# Melihat skor terbaik
print("Skor terbaik:", grid.best_score_)

# Mengevaluasi model terbaik pada data pengujian
best_model = grid.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Akurasi pada data pengujian:", accuracy)
