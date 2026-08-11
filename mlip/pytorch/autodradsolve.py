import torch,math
torch.set_default_dtype(torch.float64)

## P1

# x = torch.tensor([2.0, 3.0],requires_grad = True)
# a = x.is_leaf
# b = x.requires_grad
# c = x.grad                              #will give none coz backward() in not run
# y = x**2
# d = y.is_leaf
# e = y.requires_grad  
# f = y.grad_fn
# print(a,b,c,d,e,f)


# x1 = torch.tensor([2.0, 3.0])
# a1 = x1.is_leaf
# b1 = x1.requires_grad
# c1 = x1.grad                              #will give none coz backward() in not run
# y1 = x1**2
# d1 = y1.is_leaf
# e1 = y1.requires_grad  
# f1 = y1.grad_fn
# print(a1,b1,c1,d1,e1,f1)

# P2

r = torch.tensor([1.2],requires_grad = True)
e = 4*(r**-12 - r**-6)
a = e.grad_fn
print(a)