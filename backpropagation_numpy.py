import numpy as np

class NN:
    def __init__(self, n_i=1, n_hs=(4, 5), n_o=1):
        self.n_i = n_i
        self.n_hs = n_hs
        self.n_o = n_o

        self.w = {1: np.random.normal(size=(n_hs[0], n_i)) / np.sqrt(n_i)}
        self.b = {1: np.random.normal(size=(n_hs[0], 1)) / np.sqrt(n_i)}


        for i in range(len(n_hs)-1):
            self.w[i+2] = np.random.normal(size=(n_hs[i+1], n_hs[i])) / np.sqrt(n_hs[i+1])
            self.b[i+2] = np.random.normal(size=(n_hs[i+1], 1)) / np.sqrt(n_hs[i+1])

        self.w[len(n_hs)+1] = np.random.normal(size=(n_o, n_hs[-1])) / np.sqrt(n_o)
        self.b[len(n_hs)+1] = np.random.normal(size=(n_o, 1)) / np.sqrt(n_o)

    def fit(self, x, y, epochs=10, lr=0.01):
        for epoch in range(epochs):
            loss = 0.0
            c_w = {}
            c_b = {}
            for k, v in self.w.items():
                c_w[k] = np.zeros(v.shape)
            for k, v in self.b.items():
                c_b[k] = np.zeros(v.shape)

            for i in range(x.shape[0]):

                # Forward
                a_k, h_k = {}, {}
                h_k[0]= np.reshape(x[i], (x[i].shape[0], 1))
                for k in range(1, len(self.n_hs)+2):
                    a_k[k] = np.matmul(self.w[k], h_k[k-1]) + self.b[k]
                    h_k[k] = sigmod(a_k[k])

                y_hat = h_k[len(self.n_hs)+1]

                loss += np.sum((y_hat - y[i])) ** 2 / 2

                # --------------------------------------------------------------------------------------------
                # Back propagation
                o_e = y_hat - y[i]
                # Compute the gradient of last layer
                g = o_e
                for k in range(len(self.n_hs)+1, 0, -1):
                    g = np.multiply(g, dsigmod(a_k[k]))
                    c_w[k] = np.matmul(g, h_k[k-1].T)
                    c_b[k] = g
                    g = np.matmul(self.w[k].T, g)


            loss = loss / x.shape[0]

            for k in c_w.keys():
                c_w[k] /= x.shape[0]
                c_b[k] /= x.shape[0]

            for k in self.w.keys():
                self.w[k] -= c_w[k] * lr
                self.b[k] -= c_b[k] * lr

            print('epoch: %d, loss: %f' % (epoch, loss))

    def predict(self, x):
        a_k, h_k = {}, {}
        h_k[0]= np.reshape(x, (x.shape[0], 1))
        for k in range(1, len(self.n_hs)+2):
                    a_k[k] = np.matmul(self.w[k], h_k[k-1]) + self.b[k]
                    h_k[k] = sigmod(a_k[k])

        y_hat = h_k[len(self.n_hs)+1]

        return y_hat



def sigmod(x):
    return 1.0/(1.0+np.exp(-x))


def dsigmod(x):
    # print(x)
    y = sigmod(x)
    return y*(1-y)


if __name__ == '__main__':
    mlp = NN(1, (10, 8), 1)
    x = np.random.random(size=(100, 1))
    y = np.sin(x)
    mlp.fit(x, y, epochs=100000, lr=0.01)
    print('x = %f, predict = %f, real = %f' % (0, mlp.predict(np.array([0])), np.sin(0)))
