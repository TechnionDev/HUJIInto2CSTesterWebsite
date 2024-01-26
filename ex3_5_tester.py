#######################################################
#                  Exercise 3 Tests                   #
#                    Instructions                     #
#   1. Move this file the to exercise folder          #
#   2. Make sure there aren't any function calls      #
#      in your exercise files                         #
#   3. Run this file and check for errors or success  #
#######################################################
# Output if succeed:
# - Pycham(pytext plugin):
#     "============== 8 passed in *.**s =============="
# - CScode(default python output):
#      "Ran 8 tests in *.**s
#       OK"


import unittest
from ex3_5.ex3_5 import *


class MyTestCase(unittest.TestCase):
    def test_Q1(self):
        """test for question 1"""
        self.assertEqual([1], diagonal_sums([[1]]))
        self.assertEqual([2, 5, 3], diagonal_sums([[1, 2], [3, 4]]))
        self.assertEqual([3, 8, 15, 12, 7], diagonal_sums([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        self.assertEqual([4, 11, 21, 34, 30, 23, 13],
                         diagonal_sums([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]]))

    def test_Q2(self):
        """test for question 2"""
        self.assertEqual(True, is_submatrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]], [[1]]))
        self.assertEqual(False, is_submatrix([[1, 2, 3], [2, 'b', 6], [4, 8, 'c']],
                                             [['a', 6], [0, 'd']]))
        self.assertEqual(True,
                         is_submatrix([['a', 'b', 'c'], ['d', 'e', 'f'], ['g', 'h', 'i']],
                                      [['d', 'e'], ['g', 'h']]))
        self.assertEqual(False,
                         is_submatrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
                                      [[6, 7], [10, 12]]))
        self.assertEqual(True, is_submatrix([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]], [[6, 7, 8], [10, 11, 12]]))
        self.assertEqual(False, is_submatrix([[1, 2], [5, 6], [9, 10]], [[1, 0, 0], [0, 1, 0], [0, 0, 1]]))

    def test_Q3(self):
        """test for question 3"""
        self.assertEqual([[3, 2], [3, 2]], min_max_columns([[3, 2]]))
        self.assertEqual([[1, 2, 3], [4, 5, 6]], min_max_columns([[1, 5, 3], [4, 2, 6]]))
        self.assertEqual([[1, 2, 4], [9, 7, 8]], min_max_columns([[3, 2, 8], [1, 7, 4], [9, 6, 5]]))

    def test_Q4(self):
        """test for question 4"""
        self.assertEqual([], filter_list([], "<", 5))
        self.assertEqual([4, 5], filter_list([1, 2, 3, 4, 5], ">", 3))
        self.assertEqual([2, 2], filter_list([2, 3, 7, 4, 2], "=", 2))
        self.assertEqual([-2, -4, -3], filter_list([-2, 6, -4, 7, -3], "<", 0))

    def test_Q5(self):
        """test for question 5"""
        self.assertEqual([1], cycle_sublist([1], 0, 2))
        self.assertEqual([6, 4], cycle_sublist([6, 5, 4, 3], 0, 2))
        self.assertEqual([4, 3, 7, 6, 5], cycle_sublist([7, 6, 5, 4, 3], 3, 1))
        self.assertEqual([2, 1, 9, 3], cycle_sublist([4, 3, 2, 5, 1, 6, 9], 2, 2))
        self.assertEqual([6, 3, 1], cycle_sublist([4, 3, 2, 5, 1, 6, 9], 5, 3))

    def test_Q6a(self):
        """test for question 6 Alef"""
        self.assertEqual([[1]],
                         pascal_triangle(0))
        self.assertEqual([[1], [1, 1]],
                         pascal_triangle(1))
        self.assertEqual([[1], [1, 1], [1, 2, 1], [1, 3, 3, 1], [1, 4, 6, 4, 1]],
                         pascal_triangle(4))

    def test_Q6b(self):
        """test for question 6 Beth"""
        self.assertEqual("1", pascal_triangle_str(pascal_triangle(0)))
        self.assertEqual("_1_\n1_1", pascal_triangle_str([[1], [1, 1]]))
        self.assertEqual("__1__\n_1_1_\n1_2_1", pascal_triangle_str(pascal_triangle(2)))
        self.assertEqual("__1_2_3__\n_1_3_5_3_\n1_4_8_8_3", pascal_triangle_str(
            [[1, 2, 3], [1, 3, 5, 3], [1, 4, 8, 8, 3]]))
        self.assertEqual(
            "______1______\n_____1_1_____\n____1_2_1____\n___1_3_3_1___\n__1_4_6_4_1__\n_1_5_0_0_5_1_\n1_6_5_0_5_6_1",
            pascal_triangle_str(pascal_triangle(6)))

    def test_Q6c(self):
        """test for question 6 Gimel"""
        self.assertEqual([[1], [1, 1], [1, 2, 1], [1, 3, 3, 1]], pascal_triangle_from_base([1], 3))
        self.assertEqual([[1, 2, 3]], pascal_triangle_from_base([1, 2, 3], 0))
        self.assertEqual([[1, 2, 3], [1, 3, 5, 3], [1, 4, 8, 8, 3]], pascal_triangle_from_base([1, 2, 3], 2))
        self.assertEqual([[1, 2], [1, 3, 2], [1, 4, 5, 2], [1, 5, 9, 7, 2]], pascal_triangle_from_base([1, 2], 3))
        self.assertEqual(pascal_triangle(7), pascal_triangle_from_base([1], 7))
        base = [1, 2]
        self.assertIsNot(pascal_triangle_from_base(base, 0)[0], base, "You didn't COPY the base")


if __name__ == '__main__':
    unittest.main()
