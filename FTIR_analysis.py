# FTIR Analysis
# Ben Zager
# 

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy.io import loadmat
from scipy.linalg import solveh_banded
from scipy.signal import argrelmax
from cycler import cycler


fname = 'FTIRdata_PVA.mat'
varname = "A"
minWN = 600; maxWN = 1600; # choose range of data to consider

data = sp.io.loadmat(fname)[varname]
minIdx = np.where(np.floor(data[:,0])==minWN)[0][0] # find indices of range
maxIdx = np.where(np.floor(data[:,0])==maxWN)[0][0]

wn = data[minIdx:maxIdx,0] # reassign values from specified range
absorb = data[minIdx:maxIdx,1:] # absorbances (raw,100,110,...190)
N = absorb.shape[1] # number of samples

minT = 100; maxT = 190;
labels = [str(T)+"C" for T in xrange(minT,maxT+1,10)] # define labels
labels.insert(0,"Raw")
colorList = ['0.5','r','g','b','c','y','k','#654321','#32cd32','m','#ffa500']
plt.rc('axes',prop_cycle=(cycler('color',colorList)))


# Asymmetric least squares smoothing for baseline correction
# Written by CJ Carey, github.com/perimosocordiae
# Original paper:
# www.science.uva.nl/~hboelens/publications/draftpub/Eilers_2005.pdf
def als_baseline(intensities,asymmetry_param=0.05,smoothness_param=1e6,max_iters=10,conv_thresh=1e-5,verbose=False):
	smoother = WhittakerSmoother(intensities,smoothness_param,deriv_order=2)
	# Rename p for concision.
	p = asymmetry_param
	# Initialize weights.
	w = np.ones(intensities.shape[0])
	for i in xrange(max_iters):
		z = smoother.smooth(w)
		mask = intensities > z
		new_w = p*mask + (1-p)*(~mask)
		conv = np.linalg.norm(new_w - w)
		if verbose:
			print i+1, conv
		if conv < conv_thresh:
			break
		w = new_w
	else:
		print 'ALS did not converge in %d iterations' % max_iters
	return z

class WhittakerSmoother(object):
	def __init__(self,signal,smoothness_param,deriv_order=1):
		self.y = signal
		assert deriv_order > 0, 'deriv_order must be an int > 0'
		# Compute the fixed derivative of identity (D).
		d = np.zeros(deriv_order*2 + 1, dtype=int)
		d[deriv_order] = 1
		d = np.diff(d, n=deriv_order)
		n = self.y.shape[0]
		k = len(d)
		s = float(smoothness_param)

		# Here be dragons: essentially we're faking a big banded matrix D,
		# doing s * D.T.dot(D) with it, then taking the upper triangular bands.
		diag_sums = np.vstack([
			np.pad(s*np.cumsum(d[-i:]*d[:i]), ((k-i,0),), 'constant')
			for i in xrange(1, k+1)])
		upper_bands = np.tile(diag_sums[:,-1:], n)
		upper_bands[:,:k] = diag_sums
		for i,ds in enumerate(diag_sums):
	  		upper_bands[i,-i-1:] = ds[::-1][:i+1]
		self.upper_bands = upper_bands

	def smooth(self, w):
		foo = self.upper_bands.copy()
		foo[-1] += w  # last row is the diagonal
		return solveh_banded(foo, w * self.y, overwrite_ab=True, overwrite_b=True)


absorbCor = np.zeros([wn.size,N])

asym = 0.0001 # asymmetry parameter 
smooth = 5e5  # smoothness parameter

cor = np.zeros(absorb.shape)
for i in xrange(N):
	cor[:,i] = als_baseline(absorb[:,i],asymmetry_param=asym,smoothness_param=smooth,max_iters=50)
	absorbCor[:,i] = absorb[:,i] - cor[:,i]

# peak finding
peaks = []
for i in xrange(N):
	peak = argrelmax(absorbCor[:,i],order=20,mode='clip')[0]
	peaks.append(peak)

# plot raw data
plt.figure(1)
lines = plt.plot(wn,absorb[:,:],lw=1)
baseline = plt.plot(wn,cor[:,:],'--')
plt.xlim([maxWN,minWN])
plt.xlabel(r"Wavenumber $cm^{-1}$"); plt.ylabel("Abs");
plt.title("FTIR for Heat Treated PVA Fibers")
plt.legend(iter(lines),labels,fontsize=9,markerscale=10)

# Plot corrected data
plt.figure(2)
lines = plt.plot(wn,absorbCor[:,:],lw=1)
for i in xrange(N):
	plt.plot(wn[peaks[i][:]],absorbCor[peaks[i][:],i],'.',color=colorList[i])

plt.xlim([maxWN,minWN])
plt.xlabel(r"Wavenumber $cm^{-1}$"); plt.ylabel("Abs");
plt.title("FTIR for Heat Treated PVA Fibers")
plt.legend(iter(lines),labels,fontsize=9,markerscale=10)
plt.show()

