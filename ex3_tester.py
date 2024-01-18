import unittest
from unittest import mock
import numpy
import sys
from ex3 import ex3
import itertools

NEG_LOW = -10
HIGH = 10


class TestEx3InputList(unittest.TestCase):
    @mock.patch('ex3.ex3.input', create=True)
    def test_single_number(self, mocked_input):
        mocked_input.side_effect = ['1', '']
        res = ex3.input_list()
        self.assertEqual(res, [1.0, 1.0])

    @mock.patch('ex3.ex3.input', create=True)
    def test_empty(self, mocked_input):
        mocked_input.side_effect = ['']
        res = ex3.input_list()
        self.assertEqual(res, [0.0])

    @mock.patch('ex3.ex3.input', create=True)
    def test_multiple(self, mocked_input):
        for num in range(NEG_LOW, HIGH):
            for count in range(HIGH * 2):
                mocked_input.side_effect = [f'{i}' for i in range(num, num + count)] + ['']
                res = ex3.input_list()
                self.assertEqual(res, list(range(num, num + count)) + [sum(range(num, num + count))])


class TestEx3InnerProduct(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(0, ex3.inner_product([], []))

    def test_different_lengths(self):
        self.assertEqual(None, ex3.inner_product([1, 2, 3], []))
        self.assertEqual(None, ex3.inner_product([], [1, 2, 3]))
        self.assertEqual(None, ex3.inner_product([1, 2], [1, 2, 3]))
        self.assertEqual(None, ex3.inner_product([1, 2], [1]))

    def test_singleton(self):
        for i in range(NEG_LOW, HIGH):
            self.assertEqual(i ** 2, ex3.inner_product([i], [i]))

    def test_random_values(self):
        round_digits = 5
        rand_funcs = [numpy.random.uniform, numpy.random.randint]
        for rand_func in rand_funcs:
            for vec_len in range(2, HIGH * 10):
                # Test with different random vectors
                for _ in range(HIGH * 3):
                    vec1 = rand_func(low=NEG_LOW, high=HIGH, size=(vec_len,))
                    vec2 = rand_func(low=NEG_LOW, high=HIGH, size=(vec_len,))
                    self.assertEqual(round(numpy.inner(vec1, vec2), round_digits),
                                     round(ex3.inner_product(list(vec1), list(vec2)), round_digits))


class TestEx3Monotonicity(unittest.TestCase):
    def test_empty(self):
        self.assertEqual([True] * 4, ex3.sequence_monotonicity([]))

    def test_singleton(self):
        for i in range(NEG_LOW, HIGH):
            self.assertEqual([True] * 4, ex3.sequence_monotonicity([i]))

    def test_const(self):
        for i in range(NEG_LOW, HIGH):
            self.assertEqual([True, False, True, False], ex3.sequence_monotonicity([i] * HIGH))

    def test_mono_rising(self):
        cases = ((list(range(NEG_LOW, HIGH)), [True, True, False, False]),
                 ([2 ** x for x in list(range(HIGH))], [True, True, False, False]),
                 ([x // 2 for x in list(range(NEG_LOW, HIGH))], [True, False, False, False]),
                 )
        for case in cases:
            self.assertEqual(case[1], ex3.sequence_monotonicity(case[0]), f'Failed for sequence={case[0]}')

    def test_mono_descending(self):
        cases = ((list(reversed(list(range(NEG_LOW, HIGH)))), [False, False, True, True]),
                 (list(reversed([2 ** x for x in list(range(HIGH))])), [False, False, True, True]),
                 (list(reversed([x // 2 for x in list(range(NEG_LOW, HIGH))])), [False, False, True, False])
                 )
        for case in cases:
            self.assertEqual(case[1], ex3.sequence_monotonicity(case[0]), f'Failed for sequence={case[0]}')


def sequence_monotonicity_control(sequence):
    """
    Returns the different monotonicity stats of the sequence:
    [increasing, increasing strongly, decreasing, decreasing strongly]
    """
    if len(sequence) == 0:
        return [True] * 4

    prev = sequence[0]
    increasing = True
    increasing_strongly = True
    decreasing = True
    decreasing_strongly = True

    for num in sequence[1:]:
        if num > prev:
            decreasing = False
            decreasing_strongly = False
        elif num < prev:
            increasing = False
            increasing_strongly = False
        else:
            increasing_strongly = False
            decreasing_strongly = False
        prev = num

    return [increasing, increasing_strongly, decreasing, decreasing_strongly]


def is_possible_mono_stats_for_len_4(monotonicity_stats):
    """ Check if it's possible to get these monotonicity stats """
    if monotonicity_stats[0] >= monotonicity_stats[1] and monotonicity_stats[2] >= monotonicity_stats[3]:
        return monotonicity_stats.count(True) in [0, 1, 2]
    return False


class TestEx3MonotonicityInverse(unittest.TestCase):
    def test_monotonicity_inverse_cases(self):
        # Gets all possible 4 element arrays of True/False
        for case in itertools.product((True, False), repeat=4):
            case = list(case)
            sample = ex3.monotonicity_inverse(case)
            if is_possible_mono_stats_for_len_4(case):
                self.assertTrue(sample and len(sample) == 4, f'For case: {case} got invalid sample: {sample}')
                self.assertEqual(case, sequence_monotonicity_control(sample),
                                 f'For monotonicity stats: {case} got incorrect sample: {sample}')
            else:
                self.assertTrue(sample is None, f'Got sample for impossible case: {case} got sample: {sample}')


def conv2d(a, f):
    s = f.shape + tuple(numpy.subtract(a.shape, f.shape) + 1)
    strd = numpy.lib.stride_tricks.as_strided
    subM = strd(a, shape=s, strides=a.strides * 2)
    return numpy.einsum('ij,ijkl->kl', f, subM)


class TestEx3Convolve(unittest.TestCase):
    def test_convolve_empty(self):
        self.assertEqual(None, ex3.convolve([]), f'Convolve empty got unexpected result')

    def test_convolve_3x3(self):
        self.assertEqual([[9]], ex3.convolve([[1] * 3] * 3))

    def test_rand_matrices(self):
        conv_mat = numpy.asarray([[1] * 3] * 3)
        for height in range(3, HIGH):
            for width in range(3, HIGH):
                # Run with multiple random values
                for _ in range(HIGH * 2):
                    # Create random matrix using numpy
                    mat = numpy.random.rand(height, width).round(5)
                    expected = conv2d(mat, conv_mat).round(5)
                    actual = numpy.asarray(ex3.convolve(mat.tolist())).round(5)
                    self.assertEqual(expected.tolist(), actual.tolist())


class TestEx3SumVectors(unittest.TestCase):
    def test_sum_empty(self):
        self.assertEqual(None, ex3.sum_of_vectors([]), f'Expected None for empty list of vectors')

    def test_random_vectors(self):
        # Run the test multiple times
        for _ in range(HIGH):
            for size in range(HIGH):
                for count in range(HIGH):
                    vectors = []
                    for _ in range(count):
                        vectors.append(numpy.random.uniform(low=NEG_LOW, high=HIGH, size=(size,)))
                    if count == 0:
                        actual = ex3.sum_of_vectors(vectors)
                        expected = None
                    else:
                        actual = numpy.asarray(ex3.sum_of_vectors(vectors)).round(5).tolist()
                        expected = numpy.add.reduce(numpy.asarray(vectors)).round(5).tolist()

                    self.assertEqual(expected, actual, f'Incorrect result for vectors: {vectors}')


def num_of_orthogonal_control(vectors):
    """ Returns the number of vector pairs that are orthogonal to each other """

    vectors = numpy.asarray(vectors)
    count = 0
    for i in range(len(vectors)):
        vec1 = vectors[i]
        # Check against only superseding vectors to avoid counting vectors twice
        #                                     (as well as checking vectors against themselves)
        for vec2 in vectors[i + 1:]:
            if numpy.inner(vec1, vec2) == 0:
                count += 1

    return count


class TestEx3Orthogonal(unittest.TestCase):
    def test_empty(self):
        self.assertEqual(0, ex3.num_of_orthogonal([]))

    def test_singleton(self):
        self.assertEqual(0, ex3.num_of_orthogonal([[1]]))
        self.assertEqual(0, ex3.num_of_orthogonal([[-1]]))
        self.assertEqual(0, ex3.num_of_orthogonal([[0]]))

    def test_random_vectors(self):
        for count in range(0, HIGH * 10, 7):
            for size in range(1, HIGH):
                vectors = numpy.random.randint(low=NEG_LOW, high=HIGH, size=(count, size))
                expected = num_of_orthogonal_control(vectors)
                actual = ex3.num_of_orthogonal(vectors.tolist())
                self.assertEqual(expected, actual, f'Unexpected orthogonal count for vectors: {vectors}')


if __name__ == '__main__':
    unittest.main()
