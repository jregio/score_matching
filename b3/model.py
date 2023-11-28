import torch
import numpy as np
from matplotlib import pyplot as plt

class NeuralNetwork(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(NeuralNetwork, self).__init__()

        self.linear1 = torch.nn.Linear(input_dim, 128)
        self.activation1 = torch.nn.Softplus()

        self.linear2 = torch.nn.Linear(128, 128)
        self.activation2 = torch.nn.Softplus()

        self.linear3 = torch.nn.Linear(128, output_dim)
        self.activation3 = torch.nn.Identity()
    
    def forward(self, x):
        x = self.linear1(x)
        x = self.activation1(x)

        x = self.linear2(x)
        x = self.activation2(x)

        x = self.linear3(x)
        x = self.activation3(x)

        return x
    
class Model:
    def __init__(self, mode, distribution):
        self.distribution = distribution
        self.model = NeuralNetwork(distribution.dim, distribution.dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)
        self.loss_fn = torch.nn.MSELoss()
    
    def train(self, train_iters, batch_size):
        iter_vec = np.arange(1, train_iters + 1)
        loss_vec = np.zeros(train_iters)

        for i in range(train_iters):
            print("Iteration ", i)
            inputs, labels = self.distribution.generate_training_data('score', batch_size)
            self.optimizer.zero_grad()

            outputs = self.model.forward(inputs)
            loss = self.loss_fn(outputs, labels)
            loss_vec[i] = loss.item()
            print(loss_vec[i])

            loss.backward()
            self.optimizer.step()

        return iter_vec, loss_vec
    
    def riemann_sum(self, func, x, a, n):
        epsilon = (x - a) / n

        sum = 0
        for k in range(n):
            sum += func(a + (k * epsilon)) * epsilon
        
        return sum

    def estimate_partition(self, a, n):
        S = torch.zeros(n + 1)
        width =  ((-a - a) / n)

        S[0] = a
        for i in range(1, len(S)):
            S[i] = S[i - 1] + width
        S = S.float()

        partition = 0
        for s in S:
            partition += np.exp(self.riemann_sum(self.estimate_score, s, a, n)) * width
        
        return partition

    def estimate_score(self, x):
        return self.model.forward(torch.tensor([x])).detach()

    def estimate_logprob_unnormalized(self, x, a, n):
        return self.riemann_sum(self.estimate_score, x, a, n)

    def estimate_prob_unnormalized(self, x, a, n):
        return np.exp(self.estimate_logprob(x, a, n))

    def run(self, train_iters=6000, batch_size=1000, test_size=100, a=-6, n=120):
        iter_vec, loss_vec = self.train(train_iters, batch_size)

        self.plot_training(iter_vec, loss_vec)

        modes = ['score', 'logprob', 'prob']
        for mode in modes:
            self.plot_testing(mode, a, n)
        
        print("Expected error: ", self.calculate_expected_error(a, n))
    
    def plot_training(self, iter_vec, loss_vec):
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.plot(iter_vec, loss_vec)
        plt.savefig(str.format('plot_{dist}_training.png', dist=self.distribution.name))
        plt.close()

    def plot_testing(self, mode, a, n):
        if mode == 'score':
            func = self.estimate_score
        elif mode == 'logprob':
            partition = self.estimate_partition(a, n)
            func = lambda x: self.estimate_logprob_unnormalized(x, a, n) - np.log(partition)
        elif mode == 'prob':
            partition = self.estimate_partition(a, n)
            func = lambda x: np.exp(self.estimate_logprob_unnormalized(x, a, n) - np.log(partition))

        x  = torch.linspace(-5, 5, steps=1000)
        y_truth = self.distribution.map(mode, x)
        y_pred = [func(x[i]) for i in range(len(x))]
        
        fig, ax = plt.subplots()
        plt.xlabel('x')
        plt.ylabel(mode)
        plt.plot(x, y_truth, color='b', label='Ground truth')
        plt.plot(x, y_pred, color='r', label='Prediction')
        ax.legend()
        plt.savefig(str.format('plot_{mode}_{dist}_testing.png', mode=mode, dist=self.distribution.name))
        plt.close()

    def calculate_expected_error(self, a, n):
        width =  ((-a - a) / n)
        
        expected_error = 0
        for k in range(n):
            xk = a + k * width
            prob = self.distribution.prob(torch.tensor([xk]))
            score_label = self.distribution.score(torch.tensor([xk]))
            score_pred = self.estimate_score(xk)
            expected_error += prob * (score_label - score_pred) ** 2

        return expected_error