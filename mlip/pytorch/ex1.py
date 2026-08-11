import torch
import numpy as np

# ta = torch.tensor([1.0,2.0,3.0,15,125,22,123,33,66,98,27,25])
# npa = np.array([11,22,33])
# z = torch.zeros(3,4)
# g = torch.linspace(1,10,20)
# s = ta.sum()
# m = ta.mean()
# l = len(ta)
# n = l/3
# y = ta.reshape(int(n),3)
# print(s,m,y)
# print(y[3])
# n2t = torch.from_numpy(npa) # numpy to tensor
# t2n = n2t.numpy()           # tensor to numpy
# print(n2t,t2n)


## lj , autograd use ##
r  = torch.tensor([1.2], requires_grad = True)
e = 4.0 * (r **12 - r **6)
e.backward()
print(r.grad,e)

## torch.autograd.grad(output, inputs)
# r = torch.tensor([1.2], requires_grad=True)
# e = 4.0 * (r**-12 - r**-6)
# (dedr,) = torch.autograd.grad(e, r)     # returns a tuple
# force = -dedr                            # F = -dE/dr