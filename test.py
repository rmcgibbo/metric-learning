import numpy as np
from lmmdm import optimize_metric, optimize_dmetric
import matplotlib.pyplot as pp
import IPython as ip

def generate_data(num_data, dims):
    r1 =  np.random.randn(num_data, dims)
    r2 = np.random.randn(num_data, dims)
    r1[:,0] *= 5
    r2[:,0] *= 5
    r2[:,1] += 2
    
    return r1, r2


class Plotter(object):
    def __init__(self, figs=1):
        self.xmin, self.xmax = None, None
        self.ymin, self.ymax = None, None
        self.fig = pp.figure()
        if figs == 1:
            self.ax1 = self.fig.add_subplot(111, aspect='equal')
        if figs == 2:
            self.ax1 = self.fig.add_subplot(211, aspect='equal')
            self.ax1.set_aspect('equal')
            self.ax1.grid(True)
            self.ax2 = self.fig.add_subplot(212, aspect='equal')
            self.ax2.grid(True)
        
    def __call__(self, r1, r2, triplets=None, subfig=1):
        if triplets is None:
            triplets = []
    
        r1_x = np.ravel(r1[:,0])
        r2_x = np.ravel(r2[:,0])
        r1_y = np.ravel(r1[:,1])
        r2_y = np.ravel(r2[:,1])
        
        if subfig == 1:
            ax = self.ax1
        elif subfig == 2:
            ax = self.ax2
            
        ax.scatter(r1_x, r1_y, c='c', alpha=0.75, edgecolors='none')
        ax.scatter(r2_x, r2_y, c='m', alpha=0.75, edgecolors='none')
    
        for a,b,c in triplets:
            ab_x = [float(a[0]), float(b[0])]
            ab_y = [float(a[1]), float(b[1])]
            ac_x = [float(a[0]), float(c[0])]
            ac_y = [float(a[1]), float(c[1])]
            ax.plot(ab_x, ab_y, 'g', lw=1.5)
            ax.plot(ac_x, ac_y, 'r', lw=1.5)
        
        #if self.xmin is not None:
        #    pp.xlim(self.xmin, self.xmax)
        #    pp.ylim(self.ymin, self.ymax)

        #self.ymin, self.ymax = pp.ylim()
        #self.xmin, self.xmax = pp.xlim()

def get_triplets(r1, r2, num_triplets):
    num_data, dims = r1.shape
    assert r1.shape == r2.shape, 'same shape'
    
    # each triplet (a,b,c) should have the semantic
    # a is closer to b than a is to c
    a = np.empty((num_triplets, dims))
    b = np.empty((num_triplets, dims))
    c = np.empty((num_triplets, dims))
    
    for i in range(num_triplets):
        if np.random.randint(2) == 0:
            # pick 2 from r1 and 1 from r2
            a[i,:] = r1[np.random.randint(num_data)]
            b[i,:] = r1[np.random.randint(num_data)]
            c[i,:] = r2[np.random.randint(num_data)]
        else:
            # pick 1 from r2 and 2 from r2
            a[i,:] = r2[np.random.randint(num_data)]
            b[i,:] = r2[np.random.randint(num_data)]
            c[i,:] = r1[np.random.randint(num_data)]
        
    return (a,b,c)


def main():
    num_data = 1000
    dims = 4
    num_triplets = 200
    r1, r2 = generate_data(num_data, dims)
    triplets = get_triplets(r1, r2, num_triplets)
    triplets2 = tuple(np.copy(t) for t in triplets)
    
    #X = optimize_metric(triplets, alpha=2, beta=0)
    #print 'Full Rank'
    #print X
    X = optimize_dmetric(triplets2, alpha=2)
    print 'Diagonal'
    print X
    sys.exit(1)
    
    
    u, v = np.linalg.eig(X)
    u[np.where(u < 0)] = 0
    t = np.real(v * np.diag(np.sqrt(u))*np.linalg.inv(v))
    
    print t
    plot = Plotter(2)
    pp.title('Original Data')
    plot(r1, r2, subfig=1)
    r1_t = r1 * t
    r2_t = r2 * t
    #pp.title('Projected Data')
    plot(r1_t, r2_t, subfig=2)
    #ip.embed()
    pp.show()
    

if __name__ == '__main__':
    main()