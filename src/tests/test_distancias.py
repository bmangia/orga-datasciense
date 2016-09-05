import unittest
from distancias import *
from math import sqrt

class TestDistancias(unittest.TestCase):

	def test_distanciaEcluedianaEntreIgualesPuntosEsCero(self):
		self.assertTrue(euclediana((0,0),(0,0)) == 0)

	def test_distanciaEcluedianaEntreEsIgualA1(self):
		self.assertTrue(euclediana((1,0),(0,0)) == 1)
		self.assertTrue(euclediana((0,0),(0,1)) == 1)
		self.assertTrue(euclediana((2,0),(1,0)) == 1)
		self.assertTrue(euclediana((0,1),(0,2)) == 1)

	def test_distanciaEcluedianaEsRaizDe2(self):
		self.assertTrue(euclediana((1,0),(0,1)) == sqrt(2))

if __name__ == '__main__':
	unittest.main()