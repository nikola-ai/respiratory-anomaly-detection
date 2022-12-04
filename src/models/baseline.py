from sklearn.dummy import DummyClassifier

random_state = 42


# generates predictions by respecting the training set’s class distribution
def stratified_classifier():
    return DummyClassifier(strategy="stratified", random_state=random_state)


# always predicts the most frequent label in the training set.
def most_frequent_classifier():
    return DummyClassifier(strategy="most_frequent", random_state=random_state)


# always predicts the class that maximizes the class prior
# (like “most_frequent”) and predict_proba returns the class prior
def prior_classifier():
    return DummyClassifier(strategy="prior", random_state=random_state)


# generates predictions uniformly at random
def uniform_classifier():
    return DummyClassifier(strategy="uniform", random_state=random_state)


# always predicts a constant label that is provided by the user
# This is useful for metrics that evaluate a non-majority class
def constant_classifier(constant):
    return DummyClassifier(strategy="constant", random_state=random_state,
                           constant=constant)
