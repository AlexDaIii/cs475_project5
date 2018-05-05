import numpy as np

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
        return np.array(self._potentials2[self._n:])


class SumProduct:
    def __init__(self, p):
        self._potentials = p
        # TODO: EDIT HERE
        # add whatever data structures needed

        # since our way requires us to get a whole row or matrix at a time, we are changing stuff up
        uniary = self._potentials.return_uniary()
        binary = np.array(self._potentials._potentials2)
        print(self._potentials.return_binary())
        print(binary[self._potentials.chain_length() + 1:])

        # FORWARD PASS + BACKWARD PASS AT THE SAME TIME
        # stores all mus from forward pass, each mu is a row vector
        self.n = self._potentials.chain_length()
        self.k = self._potentials.num_x_values()

        self.forward_messages = np.zeros((self._potentials.chain_length() - 1, self._potentials.num_x_values()))
        self.backward_messages = np.zeros((self._potentials.chain_length() - 1, self._potentials.num_x_values()))

        # need to do a base case
        self.forward_messages[0, :] = np.dot(uniary[1, 1:], binary[self.n + 1, 1:, 1:])
        self.backward_messages[0, :] = np.dot(binary[2 * self.n - 1, 1:, 1:], uniary[self.n, 1:].T).T

        prob_forward = np.multiply(self.forward_messages[0, :], uniary[2, 1:])
        prob_backward = np.multiply(uniary[self.n - 1, 1:], self.backward_messages[0, :])

        for i in range(1, self.n - 1):
            # FORWARD PASS
            # at the ith row, insert the message which is just a dot product of p(xi), and p(x1+1| xi)
            self.forward_messages[i, :] = np.dot(prob_forward, binary[self.n+i+1, 1:,1:])
            prob_forward = np.multiply(self.forward_messages[i, :], uniary[i+2, 1:])

            # BACKWARD PASS
            self.backward_messages[i, :] = np.dot(binary[2 * self.n - (1+i), 1:,1:], prob_backward.T).T
            prob_backward = np.multiply(uniary[self.n - i - 1, 1:], self.backward_messages[i, :])

        # SAVE THE LAST AND FIRST NODE PROBS BC THEY ARE SPECIAL
        self.last_node_prob = np.divide(prob_forward, np.sum(prob_forward))
        self.first_node_prob = np.divide(prob_backward, np.sum(prob_backward))

        print(self.last_node_prob)
        print(self.first_node_prob)

    def marginal_probability(self, x_i):
        """
        :param x_i: The node we are trying to find the marginal probability of
        :return: a float list, length k=k+1
        """
        # TODO: EDIT HERE
        # should return a python list of type float, with its length=k+1, and the first value 0

        # This code is used for testing only and should be removed in your implementation.
        # It creates a uniform distribution, leaving the first position 0
        result = [1.0 / (self._potentials.num_x_values())] * (self._potentials.num_x_values() + 1)
        result[0] = 0
        return result


class MaxSum:
    def __init__(self, p):
        self._potentials = p
        self._assignments = [0] * (p.chain_length() + 1)
        # TODO: EDIT HERE
        # add whatever data structures needed

    def get_assignments(self):
        return self._assignments

    def max_probability(self, x_i):
        # TODO: EDIT HERE
        return 0.0
