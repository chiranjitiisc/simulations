import torch 

if torch.cuda.is_available():
    print("GPU found")
    print(f"GPU is {torch.cuda.get_device_name(0)}")
else:
    print("GPU not found")


# # creating a tensor

# a = torch.empty(2,3) # 2d tensor matrix, it allocates memory 
# print(a, type(a))

# # create zeros
# b = torch.zeros(2,3)
# print(b)

# # create ones
# c = torch.ones(2,5)
# print(c)

# # create randoms
# torch.manual_seed(0)
# d = torch.rand(2,3)
# print(d)

# # create custom tensors
# e = torch.tensor([[1,3,4],[2,5,6],[3,44,56.4]])
# print(e)

# # create range
# f = torch.linspace(0,10,11)
# print(f)
 
# # create identity matrix
# g = torch.eye(5)
# print(g)

# # full matrix
# h = torch.full((3,2),3) # elementsa with these values [(shape of matrix, e.g.2/3), value]
# print(h)

# # tensor with specific data types
# i = torch.tensor([[1.0,2.0,3.0],[2.5,4.55,6.7]],dtype = torch.float64)
# print(i)
# print(i[0])

# # convert datat type 
# j = i.to(torch.int64)
# print(j)


## mathematical operations

# x = torch.rand(3,3,dtype = torch.float64)
# y = torch.rand(3,3, dtype = torch.float64)

# z = x+y
# z1 = x@y
# z2 = -1 * x**2

# print(x)
# print(y)
# print('z =  ',z)
# print('z1 = ',z1)
# print('z2 = ',z2)
# print(abs(z2))

# clamp # makes all the values those are above max to max and same for min
# a = torch.rand(2,2)
# b = torch.clamp(a,min = 0.5, max = 0.8)
# print(a)
# print(b) 

################# reduction
# torch.manual_seed(6)
# a = 10 * torch.rand(3,3,dtype = torch.float64)
# print(a)
# b = torch.sum(a)                                       # sum of all elements of the matrix
# b1 = torch.prod(a)
# # print(b)
# c = torch.sum(a, dim=0)                                # sum along columns
# d = torch.sum(a, dim=1)                                # sum along rows
# e = torch.mean(a)                                      # mean of all
# f = torch.mean(a,dim = 0)                              # mean of col
# g = torch.mean(a,dim = 1)                              # mean of rows
# h = torch.argmax(a)                                    # position of max value in matrix
# i = torch.argmin(a)                                    # position of max 
# print(c)
# print(d)
# print(e)
# print(f)
# print(g)
# print(i)

##################### matrix ops
# a1 = torch.rand(3,3)
# a = torch.randint(size = (3,3),low = 0,high = 10)
# b = torch.randint(size = (3,3),low = 0,high = 10)
# c = torch.matmul(a,b)
# print(a,b,c)
# v1 = torch.tensor([1,2,3])
# v2 = torch.tensor([4,5,6])
# d = torch.dot(v1,v2)
# print(d)
# d2 = torch.det(a1)
# print(d2)
# e = torch.exp(a)
# f = torch.sqrt(a)
# print(a,f)
##################### activations

# m = torch.tensor([[1,2,3],[2,-3,-4],[-1.9,-2.3,98]])
# n = torch.relu(m)                                           # it applies relu to the tensor m and saves it to a new tensor named n
# k = torch.relu_(m)                                          # it applies relu to the tensor m and modifies the tensor itself now the m tensor is reluof(m)
# print(k,n,m)                                                # the _ after any fuction makes it inplace operator

########################################################################################################################################################

## copy a tensor, we dont want the sorce and copy to be modified when we make changes in the sorce so we clone it 
torch.manual_seed(0)

a = torch.rand(3,3,dtype = torch.float64)
print(a)
a = b 

b = torch.clone(a)
print(b)







 