import unittest

class PythonTestCase(unittest.TestCase):
    def test_range(self):
        for i in range(10):
            print(i)
        self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()

