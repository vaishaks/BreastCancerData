import numpy as np
from sklearn import svm
from sklearn import cross_validation as cv
from sklearn.grid_search import GridSearchCV
import matplotlib.pyplot as plt

# Extracting data from the data file into a format from which we can learn our model
data = open("data/wdbc.data")
X = []
y = []
for line in data:
    linelist = line.split(',')
    linelist[31].replace("\n", "")
    X.append(map(float, linelist[2:]))
    if linelist[1] == 'M':
        y.append(1)
    else:
         y.append(0)
data.close()

# Convert the python list into a numpy array
X = np.array(X, dtype=float)
y = np.array(y)

# Splitting data into training and test datasets
X_train, X_test, y_train, y_test = cv.train_test_split(X, y, test_size=0.2, 
                                                        random_state=0)

# Trying to find an optimal value for parameters C and gamma using GridSearch
# Range of C values - 10^-2 to 10^4
C_range = 10.0 ** np.arange(-2, 4)
# Range of gamma values - 10^-5 to 10
gamma_range = 10.0 ** np.arange(-5, 1)
# Create a parameter grid i.e. a python dict
param_grid = dict(gamma=gamma_range, C=C_range)
# Cross validation method
skf = cv.StratifiedKFold(y=y_train, n_folds=3)
grid = GridSearchCV(svm.SVC(kernel="linear"), param_grid=param_grid, cv=skf)
grid.fit(X_train, y_train)

# Train a classifier using the best parameters found using GridSearch
clf = svm.SVC(kernel="linear", **grid.best_params_)
# Mean of a five fold cross validation score
print cv.cross_val_score(clf, X_train, y_train, cv=5).mean()

# Visualizing the data using the first two dimensions
plt.scatter([X[x][0] for x in range(len(X)) if y[x] == 0], 
                [X[x][1] for x in range(len(X)) if y[x] == 0], label="Benign")
plt.scatter([X[x][0] for x in range(len(X)) if y[x] == 1], 
                [X[x][1] for x in range(len(X)) if y[x] == 1], color="red", 
                    label="Malignant")
plt.legend()
plt.xlabel("Radius")
plt.ylabel("Texture")
plt.title("Radius vs Texture")
plt.show()
