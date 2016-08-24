import numpy as np
np.random.seed(10)

def coeffs(n):
     return [np.random.uniform() for i in xrange(0, n)]

def gen_data(coeffs, num_rows, width):
    points = [[np.random.uniform(-width, width) for i in range(len(coeffs))] for i in range(num_rows)]
    vals = [sum([coeff * j for (j,coeff) in zip(i,coeffs)]) for i in points]
    
    for i, x in enumerate(points):
            print "{0},{1}".format(','.join(map(str, x)), str(vals[i]))

if __name__ == '__main__':
    cf = coeffs(100)
    gen_data(cf, 1000, 5)
