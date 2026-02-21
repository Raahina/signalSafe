# train_classifier.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def main():
    X = np.load("X_hand.npy")
    y = np.load("y_hand.npy")

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Class counts:", {c: (y == c).sum() for c in np.unique(y)})

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42
    )

    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    print("Classification report:")
    print(classification_report(y_test, y_pred))

    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    joblib.dump(clf, "distress_classifier.joblib")
    print("Saved classifier to distress_classifier.joblib")

if __name__ == "__main__":
    main()

