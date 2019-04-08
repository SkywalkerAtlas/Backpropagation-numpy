import numpy as np

class NN:
    def __init__(self, n_i=4, n_h=5, n_o=3):
        self.n_i = n_i
        self.n_h = n_h
        self.n_o = n_o

        self.w_ih = np.random.normal(size=(self.n_h, self.n_i))
        self.w_ho = np.random.normal(size=(self.n_o, self.n_h))

        self.b_ih = np.random.normal(size=(self.n_h, 1))
        self.b_ho = np.random.normal(size=(self.n_o, 1))

        # self.c_w_ih = np.zeros(shape=(self.n_h, self.n_i))
        # self.c_w_ho = np.zeros(shape=(self.n_o, self.n_h))


    def fit(self, x, y, epochs=10, lr=0.01):
        for epoch in range(epochs):
            loss = 0.0
            c_w_ih = np.zeros(shape=(self.n_h, self.n_i))
            c_w_ho = np.zeros(shape=(self.n_o, self.n_h))
            c_b_ih = np.zeros((self.n_h, 1))
            c_b_ho = np.zeros((self.n_o, 1))
            for i in range(x.shape[0]):
                a_1 = self.w_ih * x[i] + self.b_ih
                h_1 = sigmod(a_1)

                y_hat = np.matmul(self.w_ho, h_1) + self.b_ho

                loss += (y_hat - y[i]) ** 2 / 2

                # Back propagation
                o_e = y_hat - y[i]

                c_w_ho += o_e * h_1.T
                c_b_ho += o_e

                d_h = np.matmul(self.w_ho.T, o_e)
                g = np.multiply(d_h, dsigmod(h_1))
                # print(g.shape, x[i].T.shape)
                c_w_ih += g * x[i].T
                c_b_ih += g

                # self.c_w_ih += np.matmul(np.multiply(np.matmul(self.w_ho.T, o_e), dsigmod(a_1)), x[i].T)

            loss = loss / x.shape[0]
            c_w_ho, c_w_ih = c_w_ho / x.shape[0], c_w_ih / x.shape[0]
            c_b_ih, c_b_ho = c_b_ih / x.shape[0], c_b_ho / x.shape[0]

            self.w_ho -= c_w_ho * lr
            self.w_ih -= c_w_ih * lr
            self.b_ho -= c_b_ho * lr
            self.b_ih -= c_b_ih * lr

            print('epoch: %d, loss: %f' % (epoch, loss))

    def predict(self, x):
        h_1 = sigmod(self.w_ih * x)
        y_hat = np.matmul(self.w_ho, h_1)

        return y_hat



def sigmod(x):
    return 1.0/(1.0+np.exp(-x))


def dsigmod(y):
    return y*(1-y)


if __name__ == '__main__':
    mlp = NN(1, 10, 1)
    x = np.random.random(size=(100, 1))*10
    y = np.sin(x)
    mlp.fit(x, y, epochs=50000, lr=0.1)
    print('x = %f, predict = %f, real = %f' % (0, mlp.predict(np.array([0])), np.sin(0)))
