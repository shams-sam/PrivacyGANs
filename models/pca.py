from sklearn.decomposition import PCA


class PCABasic():
    def __init__(self, threshold=0.99):
        self.threshold = threshold
        self.pca = PCA()

    def train(self, input_data, num_components=False):
        self.pca.fit_transform(input_data)
        if not num_components:
            self.num_components = self.calculate_components()
        else:
            self.num_components = num_components
        self.pca = PCA(n_components=self.num_components)

        return self.pca.fit_transform(input_data)

    def eval(self, input_data):

        return self.pca.transform(input_data)

    def calculate_components(self):
        num_components = 0
        total_var = 0
        for _ in self.pca.explained_variance_ratio_:
            total_var += _
            num_components += 1
            if total_var > self.threshold:
                break

        return num_components
