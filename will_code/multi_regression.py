import matplotlib.pyplot as plt
import numpy as np
import os


# load data set and define constraint matrix and vector
H = np.loadtxt(os.path.join('DATA', 'housing.data'))
b = H[:, -1] # final column (housing values)
A = H[:, :-1] # all other columns (housing attributes)

# pad with ones for nonzero offset (adding one column of 1's at end of A)
A = np.pad(A,[(0,0),(0,1)], mode='constant', constant_values=1)

# solve Ax=b using SVD (economy)
U, S, VT = np.linalg.svd(A, full_matrices=0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

# plot the regressions results
fig = plt.figure()
ax1 = fig.add_subplot(121)

plt.plot(b, color='k', linewidth=2, label='Housing Value') # True relationship
plt.plot(A@x, '-o', color='r', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.ylabel('Median Home Value [$1k]')
plt.title("Regression Results")
plt.legend()

ax2 = fig.add_subplot(122)
sort_ind = np.argsort(H[:,-1])
b = b[sort_ind] # sorted values
plt.plot(b, color='k', linewidth=2, label='Housing Value') # True relationship
plt.plot(A[sort_ind,:]@x, '-o', color='r', linewidth=1.5, markersize=6, label='Regression')
plt.xlabel('Neighborhood')
plt.title("Sorted Regression Results")
plt.legend()

plt.show()


# find the mean of each column (and reshape to column vector)
A_mean = np.mean(A,axis=0)
A_mean = A_mean.reshape(-1, 1)

# subtract the mean from each column
A2 = A - np.ones((A.shape[0],1)) @ A_mean.T

# normalize each column by its standard deviation
for j in range(A.shape[1]-1):
    A2std = np.std(A2[:,j])
    A2[:,j] = A2[:,j]/A2std

# pad with ones for nonzero offset
A2[:,-1] = np.ones(A.shape[0])

# solve Ax=b (normalized) using SVD (economy)
U, S, VT = np.linalg.svd(A2, full_matrices=0)
x = VT.T @ np.linalg.inv(np.diag(S)) @ U.T @ b

# plot the significance (magnitude) of each attribute
x_tick = range(len(x)-1)+np.ones(len(x)-1)
plt.bar(x_tick,x[:-1])
plt.xlabel('Attribute')
plt.ylabel('Magnitude')
plt.xticks(x_tick)
plt.title("Magnitude of Each Attribute in A")
plt.show()