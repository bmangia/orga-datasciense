from math import sqrt

# p1: (x,y)
# p2: (x,y)
# return: value
def euclediana(p1, p2):
	return sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 )