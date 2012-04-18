import sys, os
import numpy as np
import warnings
from scipy.optimize import fmin_tnc, fmin_bfgs
from scipy.optimize.linesearch import line_search_wolfe1
from scipy.sparse.linalg import eigs as sparse_eigs
from scipy.sparse.linalg import eigsh as sparse_eigsh
from memoize import memoized

def get_triplets(trajectory, close, far, vector_metric, alt_metric=None, alt_cutoff=None, alt_comparator=None,
    prepared_alt_reference=None):
    
    if alt_cutoff is not None:
        assert alt_metric is not None
        assert alt_comparator is not None
        assert prepared_alt_reference is not None
        
        rmsdtraj = alt_metric.prepare_trajectory(trajectory[::close])
        a_indices = np.where(alt_comparator(alt_metric.one_to_all(prepared_alt_reference, rmsdtraj, 0),
                                            alt_cutoff))[0] * close
        b_indices = a_indices + close
        c_indices = a_indices + far
        print len(a_indices)
        
        #print 'a_indices', a_indices
        #print 'b_indices', b_indices
        #print 'c_indices', c_indices
        #print 'traj length', len(trajectory)
        
        del rmsdtraj
    else:
        a_indices = np.arange(0, len(trajectory), close)
        b_indices = a_indices + close
        c_indices = a_indices + far
    
    nskim = np.count_nonzero(c_indices >= len(trajectory))
    if nskim > 0:
        a_indices = a_indices[:-nskim]
        b_indices = b_indices[:-nskim]
        c_indices = c_indices[:-nskim]
    
    num_triplets = float(len(a_indices))
    print 'retained {0} / {1} = {2}'.format(num_triplets, len(trajectory[::close]), num_triplets/len(trajectory[::close]))
    a = vector_metric.prepare_trajectory(trajectory[a_indices])
    b = vector_metric.prepare_trajectory(trajectory[b_indices])
    c = vector_metric.prepare_trajectory(trajectory[c_indices])

    
    #print 'a_indices', a_indices
    #print 'a_indices[:-nskim]', a_indices[:-nskim]
    #print 'c_indices', c_indices
    #print 'c_indices'
    #print a.shape, b.shape, c.shape, nskim
    #sys.exit(1)
    return a, b, c
    
def print_lowprec(X, printer=sys.stdout, precision=2):
    po = np.get_printoptions()
    np.set_printoptions(precision=precision)
    print >> printer, X
    np.set_printoptions(**po)
    


class DiagonalMetricCalculator(object):
    @classmethod
    def optimize_metric(cls, triplets, alpha, verbose=True):
        m = cls(triplets, alpha, verbose)
        return m.X
    
    def test1(self):
        def test_deriv(i, dx=1e-6):
            o1, g1 = self.minus_objective_and_grad(rho_and_omegas)
            new = np.copy(rho_and_omegas)
            new[i] += dx
            o2 =  self.minus_objective_and_grad(new)[0]
            
            print 'numeric' , (o2-o1)/dx
            print 'analytic', g1[i]
            assert np.abs((o2 - o1)/dx - g1[i])/g1[i] < 1e-5
        
        rho_and_omegas = np.empty(self.dim + 1)
        rho_and_omegas[0] = np.random.randn()
        rho_and_omegas[1:] = np.random.randn(self.dim)
        
        for i in range(len(rho_and_omegas)):
            test_deriv(i)
            
        print 'Passed Test 1\n\n'
    
    def test3(self):
        def test_deriv(i, dx=1e-6):
            o1, g1 = self.minus_huber_objective_and_grad(rho_and_omegas)
            new = np.copy(rho_and_omegas)
            new[i] += dx
            o2 =  self.minus_huber_objective_and_grad(new)[0]
            
            print 'numeric' , (o2-o1)/dx
            print 'analytic', g1[i]
            assert np.abs(((o2 - o1)/dx - g1[i])/(max(g1[i], (o2-o1)/dx))) < 1e-5
            
        rho_and_omegas = np.empty(self.dim + 1)
        rho_and_omegas[0] = np.random.randn()
        rho_and_omegas[1:] = np.random.randn(self.dim)
        
        for i in range(len(rho_and_omegas)):
            test_deriv(i)
            
        print 'Passed Test 3'
    
    def test2(self):
        def test_deriv(i, dx=1e-5):
            o1, g1 = self.transformed_minus_objective_and_grad(R_and_Ws)
            new = np.copy(R_and_Ws)
            new[i] += dx
            o2 = self.transformed_minus_objective_and_grad(new)[0]
            
            numeric = (o2 - o1) / dx
            analytic = g1[i]
            print 'analytic', analytic
            print 'numeric ', numeric
            assert np.abs((numeric - analytic)/max(numeric, analytic)) < 1e-4
        
        if self.dihedral_sincos_equality_constraint:
            R_and_Ws = np.random.randn(1 + self.dim/2)
        else:
            R_and_Ws = np.random.randn(self.dim + 1)
        
        for i in range(len(R_and_Ws)):
           test_deriv(i)
        
        print 'Passed Test 2'
    
    
    def __init__(self, triplets, alpha, verbose=True, is_dihedrals=True):
        a,b,c = triplets
        self.num_triplets, self.dim = a.shape
        self.num_triplets = float(self.num_triplets)
        assert b.shape == a.shape
        assert c.shape == a.shape
        self.sq_triplets_b2c = np.square(np.matrix((a - c).T)) - np.square(np.matrix((a - b).T))
        self.K = alpha/4.0
        self.printer = sys.stdout if verbose else open('/dev/null', 'w')
        
        self.dihedral_sincos_equality_constraint = bool(is_dihedrals)
        if self.dihedral_sincos_equality_constraint:
            assert self.dim % 2 == 0, 'must be an even number'
            R_and_Ws = np.ones(self.dim/2 + 1)
        else:
            R_and_Ws = np.ones(self.dim+1)
        R_and_Ws[0] = 1
        
        obj = lambda X: self.transformed_minus_objective_and_grad(X)[0]
        grad = lambda X: self.transformed_minus_objective_and_grad(X)[1]
        
        
        # self.test1()
        #self.test2()
        #self.test3()
        #sys.exit(1)
        
        R_and_Ws = fmin_bfgs(f=obj, x0=R_and_Ws, fprime=grad, disp=False)
                
        W2 = np.square(R_and_Ws[1:])
        if self.dihedral_sincos_equality_constraint:
            W2 = np.hstack((W2, W2))
        
        self.X = np.matrix(np.diag(W2/ np.sum(W2)))
        self.rho = R_and_Ws[0]**2
        
        print >> self.printer, 'X'
        print print_lowprec(self.X, printer=self.printer, precision=2)
        print >> self.printer, 'margin: {0:5f}'.format(self.rho)
        
    
    def minus_objective_and_grad(self, rho_and_omegas):
        rho = rho_and_omegas[0]
        omegas = rho_and_omegas[1:len(rho_and_omegas)]
        assert len(omegas) == self.dim
        
        margin = (np.diag(omegas)*self.sq_triplets_b2c).sum(axis=0)
        mmargin = np.ma.masked_greater(margin - rho, 0)
        mmargin = mmargin.filled(0)
        
        print 'Classification accuracy: {0}/{1}={2:5f}'.format(np.count_nonzero(margin > 0), self.num_triplets, np.count_nonzero(margin > 0)/float(self.num_triplets))
        
        loss = np.square(mmargin).sum() / self.num_triplets
        objective = self.K * rho - loss
        
        grad = np.empty(self.dim + 1)
        grad[0] = self.K  +  2 * mmargin.sum() / self.num_triplets
        grad_omegas = -2*np.matrix(mmargin) * self.sq_triplets_b2c.T / self.num_triplets
        grad[1:self.dim+1] = grad_omegas
                        
        return -objective, -grad
        
    
    def minus_huber_objective_and_grad(self, rho_and_omegas, h=0.5):
        rho = rho_and_omegas[0]
        omegas = rho_and_omegas[1:len(rho_and_omegas)]
        assert len(omegas) == self.dim
        margin = (np.diag(omegas)*self.sq_triplets_b2c).sum(axis=0)
        margin = np.ravel(np.array(margin))
        
        margin_minus_rho = margin - rho
        greater_h = np.where(margin_minus_rho >= h)[0]
        between = np.where(np.logical_and(-h <  np.array(margin_minus_rho), margin_minus_rho < h))[0]
        less_minus_h = np.where(margin_minus_rho <= -h)[0]
        
        t1 = np.sum(np.square((h - margin_minus_rho[between]))) / (4*h*self.num_triplets)
        t2 = -np.sum(margin_minus_rho[less_minus_h]) / self.num_triplets
        loss = t1 + t2
        print 'Classification accuracy: {0}/{1} = {2:5f}'.format(np.count_nonzero(margin > 0), self.num_triplets, np.count_nonzero(margin > 0)/self.num_triplets)
        
        objective = self.K * rho - loss
        
        grad = np.empty(self.dim + 1)
        grad_rho_t1 = -(2 * h * self.num_triplets)**(-1) * np.sum(h-margin_minus_rho[between])
        grad_rho_t2 = -len(less_minus_h) / self.num_triplets
        grad[0] = self.K + grad_rho_t1 + grad_rho_t2
        
        grad_omega_t1 = (2 * h * self.num_triplets)**(-1) * np.matrix(h - margin_minus_rho[between,:]) * self.sq_triplets_b2c.T[between, :]
        grad_omega_t2 = self.sq_triplets_b2c.T[less_minus_h,:].sum(axis=0) / self.num_triplets
        grad[1:] = grad_omega_t1 + grad_omega_t2
        
        #print objective, grad
        return -objective, -grad
    
    @memoized
    def transformed_minus_objective_and_grad(self, R_and_Ws):
        rho = R_and_Ws[0]**2
        W = R_and_Ws[1:]
        W2 = np.square(W)
        omegas = W2 / np.sum(W2)
        
        if self.dihedral_sincos_equality_constraint:
            assert len(W) == self.dim / 2, 'len(W)={0}, self.dim={1}'.format(len(W), self.dim)
            W = np.hstack((W, W))
            W2 = np.hstack((W2, W2))
            omegas = np.hstack((omegas, omegas))
        
        objective, grad_rho_omegas = self.minus_huber_objective_and_grad(np.hstack(([rho], omegas)))
        
        partial_R = 2*R_and_Ws[0]*grad_rho_omegas[0]
        grad_omegas = grad_rho_omegas[1:]
        
        jacobian = -2 * np.matrix(W).T * np.matrix(W2) / np.square(np.sum(W2))
        jacobian += np.diag(2*W / np.sum(W2))
        grad_W = np.array(jacobian * np.matrix(grad_omegas).T)[:,0]
        
        if self.dihedral_sincos_equality_constraint:
            grad_W = 2 * (grad_W[0:len(grad_W)/2] + grad_W[len(grad_W)/2:])
                
        #print 'Computuing: Omegas', omegas
        #print 'gradient', np.hstack(([partial_R], grad_W))
        return objective, np.hstack(([partial_R], grad_W))
    


class MetricCalculator(object):
    @classmethod
    def optimize_metric(cls, triplets, num_outer=10, num_inner=10, alpha=1, beta=0, epsilon=1e-5, verbose=True):
        """"Optimize a Large-Margin Mahalanobis Distance Metric with triplet training examples.
        
        triplets: 3 element tuple (a,b,c). Each of a,b,c should be 2D arrays of 
            length equal to the number of training examples, and width the number
            of dimensions. Each training example is the statement that a[i,:] is closer
            to b[i,:] than it is to c[i,:]
            
        num_outer: number of outer iterations of the algorithm (int)
        num_inner: number of inner iterations of the algorithm (int)
        alpha: Algorithmic parameter which trades off between loss on the training
           examples and the margin. When alpha goes to infinity, the objective function
           contains only the margin. When alpha goes to zero, the loss function is
           only penalty terms.
        beta: Regularization strength on the frobenius norm of the metric matrix.
        epsilon: Convergence cutoff. Interpreted as the percent change."""
        
        m = cls(triplets, num_outer, num_inner, alpha, beta, epsilon, verbose)
        return m.X
    
    def __init__(self, triplets, num_outer, num_inner, alpha, beta, epsilon, verbose):
        a, b, c = triplets
        self.num_triplets, self.dim = a.shape
        assert b.shape == a.shape
        assert c.shape == a.shape
        self.tripletsa2b = np.matrix((a - b).T)
        self.tripletsa2c = np.matrix((a - c).T)
        
        
        # gxm is a num_triplets length array. Each element is a matrix giving
        # the elementwise partial derivative with respect to X of the margin
        # on that training example.
        self.gxm = np.empty(self.num_triplets, dtype='object')
        for i in range(self.num_triplets):
            self.gxm[i] = self.tripletsa2c[:, i] * self.tripletsa2c[:, i].T - \
                        self.tripletsa2b[:, i] * self.tripletsa2b[:, i].T
        
        self.num_outer = num_outer
        self.num_inner = num_inner
        self.epsilon = epsilon
        self.K = alpha
        self.R = beta
        self.printer = sys.stdout if verbose else open('/dev/null', 'w')
        
        # use DiagonalMetricCalculator as initial value
        dmc = DiagonalMetricCalculator(triplets, alpha, verbose=False)
        initial_X = dmc.X
        #X = np.matrix(np.eye(self.dim)) / float(self.dim)
        
        self.X = self._outer_loop(initial_X)
    
    def _outer_loop(self, initial_X):
        """Main loop that constructs the X matrix, and iterates maximizing rho
        and the maximizing X"""
        
        X = initial_X
        prev_obj = -np.inf
        for i in range(self.num_outer):
            rho = self._find_rho(X)
            
            obj = -self._minus_objective_and_grad_rho(rho, X)[0]
            print >> self.printer, 'Iteration: {0}'.format(i)
            print >> self.printer, 'rho:       {0:5f}'.format(rho)
            #print >> self.printer, 'objective: {0:5f}'.format(obj)            
            print >> self.printer, 'X'
            import scipy.sparse
            print_lowprec(scipy.sparse.triu(X).todense(), precision=2, printer=self.printer)
            
            print >> self.printer
            
            X, finished = self._find_X(rho, X)
            if finished or np.abs((obj - prev_obj) / obj) < self.epsilon:
                break
                
            prev_obj = obj
        
            
        print >> self.printer, '\nFinal Metric:'
        print >> self.printer, 'eigvals', np.real(np.linalg.eigvals(X))
        print >> self.printer, 'Contributions to the objective function'
        m, l, r = self._objective_function_contributions
        print >> self.printer, '(alpha)*rho:    {0:5f}'.format(m)
        print >> self.printer, 'Hinge Loss:     {0:5f}'.format(l)
        print >> self.printer, 'Regularization: {0:5f}'.format(r)
        return X
    
    def _find_X(self, rho, X1):
        """Find X that maximizes the objective function at fixed rho"""
        
        objective = lambda X: self._minus_objective_and_grad_rho(rho, as_matrix(X))[0]
        gradient = lambda X: as_vector(self._minus_grad_X_objective(rho, as_matrix(X)))
        as_vector = lambda X: np.reshape(np.array(X), self.dim * self.dim)
        as_matrix = lambda V: np.matrix(np.reshape(V, (self.dim, self.dim)))
        
        #print >> self.printer, 'alg2 starting', objective(X1)
        for i in range(self.num_inner):
            current_grad = self._minus_grad_X_objective(rho, X1)
            try:
                u, v = sparse_eigsh(-current_grad, k=1, which='LA')
            except Exception as e:
                print >> self.printer, 'Warning: Sparse solver failed'
                u, v = np.linalg.eigh(-current_grad)
            u = np.real(u)[0]
            
            v = np.matrix(np.real(v))
            p = (v * v.T - X1)
            
            if u < 0:
                #print >> self.printer, 'u < 0', u
                break
            
            try:
                u2, v2 = sparse_eigsh(X1, k=1, which='LM', sigma=1)
                #u2 = np.linalg.eigvals(X1)
            except NotImplementedError:
                warnings.warn("Warning: Your sparse eigensolver does not support shift-invert mode")
                u2, v2 = sparse_eigsh(X1, k=1, which='LM')
            except Exception as e:
                print >> self.printer, 'Warning: Sparse solver failed'
                u2 = np.linalg.eigvals(X1)
            
            
            u2 = np.real(u2)[0]
            if u2 > 1 + self.epsilon:
                print >> self.printer, 'u2 > 1', u2
                break
            
            stp, f_count, g_count, f_val, old_fval, gval = line_search_wolfe1(f=objective, fprime=gradient, xk=as_vector(X1), pk=as_vector(p))
            #print 'stp', stp
            if stp == None:
                #print >> self.printer, 'breaking for stp=None'
                break
                
            if np.abs((f_val - old_fval) / old_fval) < self.epsilon:
                #print >> self.printer, 'breaking for insufficient gain'
                break
                
            X1 = X1 + stp * p
        
        #print >> self.printer, 'j: {0}'.format(i)
        return X1, i == 0
    
    def _find_rho(self, X):
        """Find rho that maximizes the objective function at fixed X"""
        # argmax with respect to p of f(X, p)
        # f(X,p) = p - C * sum_r loss <A,X> - p
        result = fmin_tnc(self._minus_objective_and_grad_rho, (0,), disp=0, args=(X,), bounds=[(0,None)])
        return result[0][0]
    
    def _minus_objective_and_grad_rho(self, rho, X):
        "compute the object function f(X, rho)"
        # sometimes rho is a 1element vector (when called from fmin_tnc)
        if not np.isscalar(rho):
            rho = rho[0]
        
        m1 = np.array(np.multiply(self.tripletsa2b, X*self.tripletsa2b)).sum(axis=0)
        m2 = np.array(np.multiply(self.tripletsa2c, X*self.tripletsa2c)).sum(axis=0)
        m = np.ma.masked_greater((m2 - m1) - rho, 0)
        
        #print 'm', m
        #print 'X'
        #print X
        
        
        if np.count_nonzero(m.mask) == self.num_triplets:
            avg_loss = 0
            grad_avg_loss = 0
        else:
            avg_loss = np.square(m).sum() / self.num_triplets
            grad_avg_loss = -2 * np.sum(m) / self.num_triplets
        
        regularization = 0.5 * self.R * np.sum(np.square(X)) if self.R != 0 else 0
        
        # record the contributions to the objective function
        self._objective_function_contributions = (self.K * rho, -avg_loss, -regularization)
        #print >> self.printer, self._objective_function_contributions 
        
        objective = self.K * rho - avg_loss - regularization
        grad = self.K - grad_avg_loss
        
        return (-objective, (-grad, ))
    
    def _minus_grad_X_objective(self, rho, X):
        """Compute the negitive elementwise partiaul of the objective function
        with respect to the X matrix"""
        # for each of the training examples where the margin - rho is negative,
        # we take 2 times the difference times the gradient with respect to X of
        # the margin and then sum them up
        m1 = np.array(np.multiply(self.tripletsa2b, X*self.tripletsa2b)).sum(axis=0)
        m2 = np.array(np.multiply(self.tripletsa2c, X*self.tripletsa2c)).sum(axis=0)
        m = np.ma.masked_greater((m2 - m1) - rho, 0)
        
        if np.count_nonzero(m.mask) == self.num_triplets:
            avg_grad_triplets = 0
        else:
            avg_grad_triplets = (2 * m * np.ma.array(self.gxm, mask=m.mask)).sum()
            avg_grad_triplets /= self.num_triplets
        
        regularization_grad = self.R * X
        
        return avg_grad_triplets - regularization_grad


optimize_metric = MetricCalculator.optimize_metric
optimize_dmetric = DiagonalMetricCalculator.optimize_metric
