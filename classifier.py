import numpy as np
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.externals import joblib
import pickle
import time


def train(car_features, noncar_features):

    # Combine features for training
    X = np.vstack((car_features, noncar_features)).astype(np.float64)
    # Fit a per column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)

    # Define the labels vector (1 for cars, 0 for noncars)
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(noncar_features))))

    # Split data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

    # Train using linear SVC
    svc = LinearSVC()
    # Log training time
    t = time.time()
    # Fit classifier to data
    svc.fit(X_train, y_train)
    # Get test accuracy
    acc = round(svc.score(X_test, y_test), 4)
    # Return classifier and test accuracy
    return svc, acc, X_scaler


if __name__=='__main__':

    # Load features data
    car_features = pickle.load(open('car_features.p', 'rb'))
    noncar_features = pickle.load(open('noncar_features.p', 'rb'))

    # Log training time
    t = time.time()
    # Train classifier on car and noncar features
    svc, acc, scaler = train(car_features, noncar_features)
    # Print training time
    print(round(time.time()-t, 2), 'Seconds to train SVC...')
    # Print test accuracy
    print('Test Accuracy of SVC:', acc)

    # Save classifier for future use/making predictions
    joblib.dump(svc, 'classifier.pkl')
    pickle.dump(scaler, open('scaler.p', 'wb'))

