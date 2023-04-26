class DiagnosisEncoder:
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        X_copy["diagnosis"] = X_copy["diagnosis"].map({"M": 1, "B": 0})
        return X_copy.values


class DataFrameSelector:
    def __init__(self, columns, input_df):
        self.columns = columns
        self.column_indices = [input_df.columns.get_loc(col) for col in columns]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[:, self.column_indices]
