# Lecture video 01
## History of pytorch

**-** `pytorch` is a deep learning library
**-** `Torch` was a library used to make tensor based calculation and was originaly written in `LUA`
**-** `pytorch` supports dynamic computation graph. means the graph can me modified during runtime.
**-** `TorchScript` can be used to serialization, means the models (build on `pytorch`) can be  used in anywhere that has no `pytorch` like `android`.
**-** `ONNX` also supported in `pytorch` (open source nural network exchange, interoperability between different library)
## Core features
**-** `pytorch` is good at tensor calculations.
**-** `pytorch` tensor computations can be done in GPU.
**-** `pytorch` Makes a dynamic computation graph.
**-** `pytorc` Makes Automatic Differentation using `autograd` module.
**-** `pytorc` can be used for distributed training.

|Aspect|PyTorch|TensorFlow|Verdict|
|------|-------|----------|-------|
|Programing language|python|Not specific|TF is better|
|Deployment|Can be used in  crossplatforms|Mature |TF is better|
|LLM|Used more|Less, More industry based|PT is better

## Core PyTorch module
|Module|Description|
|-|-|
|`torch`|the core module|
|`torch.autograd`|automatic differntation|
|`torch.nn`|includes library, activation function, loss functions and utilities to build deep learning models|
|`torch.optim`|contains optimization algorithms like SGD, Adam and RMSprop used for training neural networks|
|`torch.utils.data`|utilids for loading data|
|`torch.cuda`|for gpu training,   etc......|




