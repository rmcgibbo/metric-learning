import scipy.io
import numpy as np
import matplotlib.pyplot as pp
from Emsmbuilder import arglib
parser = arglib.ArgumentParser(description='Plot')
parser.add_argument('metric')
args = parser.parse_args()

m = scipy.io.mmread(args.metric).diagonal()
pp.plot(1+np.arange(35), m[0:35], 'bx-', label='phi')
pp.plot(1+np.arange(35), m[35:70], 'gx-', label='psi')
pp.legend(loc=2)
pp.grid()
pp.title(args.metric)
pp.savefig(args.metric + '.png')
print 'saved {0}.png'.format(args.metric)
