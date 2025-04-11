import unittest
from sequoia_tests import tests


if __name__ == '__main__':
    t = tests.TestMergeCsv()
    t.check_duplicates_test()
    t.handle_duplicates_test()