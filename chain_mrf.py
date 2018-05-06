import numpy as np
import numpy.matlib as mat
import pdb

class ChainMRFPotentials:
    def __init__(self, data_file):
        with open(data_file) as reader:
            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")
                try:
                    self._n = int(split_line[0])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                try:
                    self._k = int(split_line[1])
                except ValueError:
                    raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                break

            # create an "(n+1) by (k+1)" list for unary potentials
            self._potentials1 = [[-1.0] * ( self._k + 1) for n in range(self._n + 1)]
            # create a "2n by (k+1) by (k+1)" list for binary potentials
            self._potentials2 = [[[-1.0] * (self._k + 1) for k in range(self._k + 1)] for n in range(2 * self._n)]

            for line in reader:
                if len(line.strip()) == 0:
                    continue

                split_line = line.split(" ")

                if len(split_line) == 3:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    if i < 1 or i > self._n:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a: " + str(a))
                    if self._potentials1[i][a] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials1[i][a] = float(split_line[2])
                elif len(split_line) == 4:
                    try:
                        i = int(split_line[0])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[0] + " to integer.")
                    try:
                        a = int(split_line[1])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[1] + " to integer.")
                    try:
                        b = int(split_line[2])
                    except ValueError:
                        raise ValueError("Unable to convert " + split_line[2] + " to integer.")
                    if i < self._n + 1 or i > 2 * self._n - 1:
                        raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
                    if a < 1 or a > self._k or b < 1 or b > self._k:
                        raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
                    if self._potentials2[i][a][b] >= 0.0:
                        raise Exception("ill-formed energy file: duplicate keys: " + line)
                    self._potentials2[i][a][b] = float(split_line[3])
                else:
                    continue

            # check that all of the needed potentials were provided
            for i in range(1, self._n + 1):
                for a in range(1, self._k + 1):
                    if self._potentials1[i][a] < 0.0:
                        raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a))
            for i in range(self._n + 1, 2 * self._n):
                for a in range(1, self._k + 1):
                    for b in range(1, self._k + 1):
                        if self._potentials2[i][a][b] < 0.0:
                            raise Exception("no potential provided for i=" + str(i) + ", a=" + str(a) + ", b=" + str(b))

    def chain_length(self):
        return self._n

    def num_x_values(self):
        return self._k

    def potential(self, i, a, b = None):
        if b is None:
            if i < 1 or i > self._n:
                raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
            if a < 1 or a > self._k:
                raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a))
            return self._potentials1[i][a]

        if i < self._n + 1 or i > 2 * self._n - 1:
            raise Exception("given n=" + str(self._n) + ", illegal value for i: " + str(i))
        if a < 1 or a > self._k or b < 1 or b > self._k:
            raise Exception("given k=" + str(self._k) + ", illegal value for a=" + str(a) + " or b=" + str(b))
        return self._potentials2[i][a][b]

    def return_uniary(self):
        return np.array(self._potentials1)

    def return_binary(self):
        return np.array(self._potentials2)


class SumProduct:
    def __init__(self, p):
        self._potentials = p

        # since our way requires us to get a whole row or matrix at a time, we are changing stuff up
        self.uniary = self._potentials.return_uniary()
        self.binary = self._potentials.return_binary()

        self.n = self._potentials.chain_length()
        self.k = self._potentials.num_x_values()

        # stores all mus from forward pass, each mu is a row vector
        self.forward_messages = np.zeros((self.n - 1, self.k))
        self.backward_messages = np.zeros((self.n - 1, self.k))

        # other vars we need
        self.z = None
        self.last_node_prob = None
        self.first_node_prob = None

        # now calculate the messages
        prob_forward, prob_backward = self.calculate_msg()
        self.font_and_back_prob(prob_forward, prob_backward)

    def calculate_msg(self):
        """
        Calculates all the forward and backwards probabilities
        :return: the forward and backward unnormalized probabilities
        """
        # need to do a base case
        self.forward_messages[0, :] = np.dot(self.uniary[1, 1:], self.binary[self.n + 1, 1:, 1:])
        self.backward_messages[0, :] = np.dot(self.binary[2 * self.n - 1, 1:, 1:], self.uniary[self.n, 1:].T).T

        prob_forward = np.multiply(self.forward_messages[0, :], self.uniary[2, 1:])
        prob_backward = np.multiply(self.uniary[self.n - 1, 1:], self.backward_messages[0, :])

        # FORWARD PASS + BACKWARD PASS AT THE SAME TIME
        for i in range(1, self.n - 1):
            # FORWARD PASS
            # at the ith row, insert the message which is just a dot product of p(xi), and p(x1+1| xi)
            self.forward_messages[i, :] = np.dot(prob_forward, self.binary[self.n + i + 1, 1:, 1:])
            prob_forward = np.multiply(self.forward_messages[i, :], self.uniary[i + 2, 1:])

            # BACKWARD PASS
            self.backward_messages[i, :] = np.dot(self.binary[2 * self.n - (1 + i), 1:, 1:], prob_backward.T).T
            prob_backward = np.multiply(self.uniary[self.n - i - 1, 1:], self.backward_messages[i, :])

        return prob_forward, prob_backward

    def font_and_back_prob(self, prob_forward, prob_backward):
        """
        Calculate the front and back probabilities
        :param prob_forward: the forward unnormalized probability
        :param prob_backward: the backward unnormalized probability
        """
        # store these for the max sum algorithm
        self.z = np.sum(prob_forward)
        # SAVE THE LAST AND FIRST NODE PROBS BC THEY ARE SPECIAL
        self.last_node_prob = np.divide(prob_forward, self.z)
        self.first_node_prob = np.divide(prob_backward, self.z)
        pass

    def marginal_probability(self, x_i):
        """
        :param x_i: The node we are trying to find the marginal probability of - an int
        :return: return a python list of type float, with its length=k+1, and the first value 0
        """
        # have to make it one bigger
        result = np.zeros((1, self.k + 1))
        # if first node
        if x_i == 1:
            result[0, 1:] = self.first_node_prob
        # if last node
        elif x_i == self.n:
            result[0, 1:] = self.last_node_prob
        # if any node in the middle
        else:
            result[0, 1:] = np.multiply(self.uniary[x_i,1:], self.forward_messages[x_i - 2, :])
            result[0, 1:] = np.multiply(result[0, 1:], self.backward_messages[self.n - 1 - x_i, :])
            result[0, 1:] = np.divide(result[0, 1:], self.z)
        return result[0].tolist()


class MaxSum(SumProduct):
    def __init__(self, p):
        super().__init__(p)

        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        # TODO: EDIT HERE
        # add whatever data structures needed
        # this is just to get the Zs

        self.forward_messages_max = np.zeros((self.n + 1, self.k))
        self.backward_messages_max = np.zeros((self.n + 1, self.k))
        self.max_values = np.zeros(self.n + 1)

        self.calculate_msg_max()

    def calculate_msg_max(self):
        """
        A function exists bc of a failure of my oop skills
        """

        # FORWARD PASS
        # loops through the things in the uniary - horiz
        for num_node in range(1, self.n + 1):
            # loops through the things in the binary
            for k in range(1, self.k + 1):
                if num_node == 1:
                    self.forward_messages_max[num_node, k-1] = 0
                else:
                    pre1 = self.forward_messages_max[num_node - 1, :]
                    pre2 = pre1 + np.log(self.binary[self.n + num_node - 1, 1:, k])
                    pre3 = pre2 + np.log(self.uniary[num_node - 1, 1:])
                    self.forward_messages_max[num_node, k-1] = np.max(pre3)

        for num_node in range(self.n, 0, -1):

            for k in range(1, self.k +1):
                if num_node == self.n:
                    self.backward_messages_max[num_node, k-1] = 0
                else:
                    post1 = self.backward_messages_max[num_node + 1, :]
                    post2 = post1 + np.log(self.uniary[num_node + 1, 1:])
                    post3 = post2 + np.log(self.binary[self.n + num_node, k, 1:])
                    self.backward_messages_max[num_node, k-1] = np.max(post3)

        for i in range(1, self.n+1):
            total = self.backward_messages_max[i, :] + self.forward_messages_max[i, :] + np.log(self.uniary[i, 1:])
            self._assignments[i] = np.argmax(total) + 1
            self.max_values[i] = np.max(total) - np.log(self.z)

        # this is the fi the message from factor to xi
        #fi[0, :] = np.log(self.uniary[self.n, 1:])

        # normalize it with z
        #self.log_prob = fi - np.log(self.z)
        # print(np.max(self.log_prob))
        #assignment = np.argmax(self.log_prob)

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        # TODO: EDIT HERE
        return np.max(self.max_values[x_i])
