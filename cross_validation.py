"""
Run cross-validation to choose the parameter alpha

From running on this data:

r1 =  np.random.randn(num_data, dims)
r2 = np.random.randn(num_data, dims)
r1[:,0] *= 10
r2[:,0] *= 10
r2[:,1] += 4

The observation is that the choice of alpha doesn't really matter much.
Between alpha=0.01 and ~100, you get about the same performance. It starts
to drop off at alpha ~= 1000. My thought is that alpha should default to one,
and I'm not sure that it really shouldn't be hardcoded at one.

I guess I should confirm that this insensitivity to alpha occurs on other data
as well.

"""



from lmmdm import optimize_metric, _MetricCalculator
import numpy as np
import itertools
import matplotlib.pyplot as pp

def flatten(list_of_lists):
    return list(itertools.chain.from_iterable(list_of_lists))
    
def remove(list, item):
    list.remove(item)
    return list
    
def cross_validate(triplets, alphas, fold=5, num_outer=10, num_inner=10, epsilon=1e-5):
    
    # these list comprehensions build the training and test sets. There are *fold*
    # training sets, each of which contain (*fold* - 1) / (*fold*) of the data
    # and the fraction thats left out of the ith training set is the ith test
    # set
    sections = np.array([triplets[i::fold] for i in range(fold)], dtype='object')
    train_indx = [remove(range(fold), i) for i in range(fold)]
    test_indx = range(fold)
    train_data = np.array([flatten(sections[train_indx[i]]) for i in range(fold)], dtype='float')
    test_data = np.array([list(sections[test_indx[i]]) for i in range(fold)], dtype='float')
        
    def score_alpha(alpha):
        score = 0
        for i in range(fold):
            print '{i} of {fold}: alpha={alpha}'.format(i=i, fold=fold, alpha=alpha)
            #X = optimize_metric(train_data[i], alpha=alpha, num_outer=num_outer,
            #                    num_inner=num_inner, epsilon=epsilon, verbose=False)
            m = _MetricCalculator(train_data[i], alpha=alpha, beta=0, num_outer=num_outer,
                              num_inner=num_inner, epsilon=epsilon, verbose=False)
            #print m._objective_function_contributions
            #print m.X
            acc = accuracy(test_data[i], m.X)
            #print acc
            
            score += acc
        return score
    
    return [score_alpha(e) for e in alphas]
            
def accuracy(triplets, X):
    score = 0
    for a,b,c in triplets:
        a2b = a-b
        a2c = a-c
        val = (a2c.T * X * a2c - a2b.T * X * a2b)[0,0]
        if val > 0:
            score += 1.0
    return score



if __name__ == '__main__':
    from lmmdm import generate_data, get_triplets
    num_data, num_triplets, dims, = 1000, 200, 3
    
    for i in range(5):
        r1, r2 = generate_data(num_data, dims)
        triplets = get_triplets(r1, r2, num_triplets)
        alphas = np.logspace(-2, 3, 5)
        scores = cross_validate(triplets, alphas)
    
        pp.plot(alphas, scores, '-o')
    
    pp.xscale('log')
    pp.xlim(5e-3, 5e3)
    pp.show()