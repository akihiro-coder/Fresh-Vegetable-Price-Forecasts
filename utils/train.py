def train():
    model = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
    model.fit(X_train, y_train)
