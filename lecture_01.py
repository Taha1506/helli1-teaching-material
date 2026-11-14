import torch
from edtrace import text, image, link


def main():
    introduction()
    pytorch_motivation()
    array_representation()


def introduction():
    text("# Deep Learning with PyTorch")
    text("### From mathematical foundations to practical implementations")
    text("Welcome! 👋")

    text("## About Me")
    text("- Taha Akbari")
    text("- MSc Student in Computer Engineering at Sharif University of Technology")
    text("- Data Scientist at Torob")
    text("- Silver Medal in National Olympiad in Mathematics")

    text("## What Will We Learn?")
    text("### Part 1: Deep Learning Fundamentals")
    text("- What is deep learning, and why does it work?")
    text("- Tensors and multidimensional arrays")
    text("- Neural networks and computational graphs")
    text("- Gradients, backpropagation, and optimization")

    text("### Part 2: PyTorch Fundamentals")
    text("- Creating and manipulating tensors")
    text("- Memory layout, shapes, and strides")
    text("- Automatic differentiation with `torch.autograd`")
    text("- Building neural networks with `torch.nn`")
    text("- Training loops, loss functions, and optimizers")
    text("- Working with CPUs and GPUs")

    text("### Part 3: Problem Solving with PyTorch")
    text("- Implementing neural networks from scratch")
    text("- Training and evaluating models")
    text("- Debugging common training problems")
    text("- Writing efficient and readable PyTorch code")

    text("## If Time Permits...")
    text("- **torchvision:** Image datasets, transforms, and pretrained models")
    text("- **transformers:** Working with modern transformer architectures")
    text("- **sentence-transformers:** Text embeddings and semantic similarity")

    text("## Prerequisites")
    text("- Basic Python programming")
    text("- Familiarity with vectors, matrices, and linear algebra")
    text("- Basic calculus is helpful")
    text("- No previous PyTorch experience required!")

    text("## By the End of This Lecture...")
    text("You should be able to:")
    text("- Understand the role of pytorch in Deep Learning.")
    text("- Understand representation of multi dimensional arrays.")


def pytorch_motivation():
    text("# Why PyTorch instead of NumPy?")
    text("NumPy provides efficient multidimensional arrays and numerical operations.")
    text("PyTorch provides two particularly useful features for deep learning:")
    text("- **Automatic differentiation:** Computing gradients automatically.")
    text("- **GPU acceleration:** Performing large numerical computations efficiently.")

    text("## 1. Automatic Differentiation")
    text("Consider the following function:")
    text(r"$$f(x) = x^2 + 3x + 1$$")
    text("Its derivative is:")
    text(r"$$f'(x) = 2x + 3$$")
    text("But what if our function contains millions of operations?")
    text("Computing derivatives manually becomes impractical.")

    text("## PyTorch's Solution: Autograd")
    text("We only define the forward computation!")

    x = torch.tensor(2.0, requires_grad=True)  # @inspect x
    y = x**2 + 3*x + 1  # @inspect y

    y.backward()
    gradient = x.grad  # @inspect gradient

    text("PyTorch automatically computes the backward pass.")
    text(r"$$f'(2) = 7$$")
    text("This is essential for training deep learning models.")

    text("## 2. GPU Acceleration")
    text("Modern deep learning involves enormous amounts of numerical computation.")
    text("GPUs can execute many arithmetic operations in parallel.")
    text("This makes them particularly effective for large matrix operations.")

    text("## Moving Computations to the GPU")
    text("PyTorch makes it easy to run tensor operations on different devices.")

    device = "cuda" if torch.cuda.is_available() else "cpu"  # @inspect device

    A = torch.randn(1000, 1000, device=device)
    B = torch.randn(1000, 1000, device=device)
    C = A @ B

    text("The same matrix multiplication code runs on CPU or GPU.")
    text("We only need to choose the appropriate device.")
    text("GPU acceleration can provide substantial speedups for large workloads.")
    text("For small operations, CPU execution may actually be faster.")

    text("## Why Is PyTorch Great for Deep Learning?")
    text("- Define how inputs are transformed into outputs (forward pass).")
    text("- Automatically compute gradients (backward pass).")
    text("- Execute large tensor computations efficiently on GPUs.")

    text("**We focus on defining the computation; PyTorch handles differentiation and device execution.**")


def array_representation():
    text("# How are multidimensional arrays represented?")
    text("Consider the following matrix:")
    text(r"$$A = \begin{bmatrix}1 & 2 & 3 \\ 4 & 5 & 6\end{bmatrix}$$")
    text("How can we represent this matrix in computer memory?")

    text("## Approach 1: Contiguous arrays")
    text("In C, we can declare a multidimensional array:")
    text("`int A[2][3] = {`")
    text("`    {1, 2, 3},`")
    text("`    {4, 5, 6}`")
    text("`};`")

    image("images/contiguous_array.png", width=700)

    text("- All six elements occupy consecutive memory locations.")
    text("- C uses **row-major order**.")
    text("- This representation requires only one allocation.")
    text("- Accessing an element requires simple address arithmetic.")
    text(r"$$\operatorname{offset}(i,j) = 3i + j$$")

    text("## Approach 2: Array of pointers")
    text("Alternatively, we can allocate each row independently:")
    text("`int *A[2];`")
    text("`A[0] = malloc(3 * sizeof(int));`")
    text("`A[1] = malloc(3 * sizeof(int));`")

    image("images/pointer_rows.png", width=700)

    text("- Each row may occupy a different memory location.")
    text("- Accessing an element requires following a row pointer.")
    text("- Each row can have a different length.")
    text("- Separate allocations introduce overhead and reduce locality.")

    text("## Which representation is preferable?")
    text("For dense numerical computations, contiguous storage is usually preferable.")
    text("- Better spatial locality and cache utilization.")
    text("- No pointer chasing between rows.")
    text("- Easier vectorization and efficient bulk memory transfers.")
    text("- Particularly useful for CPU and GPU computations.")

    text("# From C arrays to PyTorch tensors")
    text("PyTorch tensors typically use a flat underlying storage.")
    text("But how can a flat array represent an arbitrary multidimensional tensor?")

    text("## Shape and strides")
    text("Two essential pieces of metadata are:")
    text("- **Shape:** Number of elements along each dimension.")
    text("- **Strides:** Number of storage elements to skip when advancing one index along each dimension.")

    A = torch.tensor([[1, 2, 3], [4, 5, 6]])  # @inspect A

    shape = A.shape  # @inspect shape
    strides = A.stride()  # @inspect strides

    text("The shape is `(2, 3)` and the strides are `(3, 1)`.")
    text("Moving down one row skips 3 elements.")
    text("Moving right one column skips 1 element.")
    text(r"$$\operatorname{offset}(i,j) = 3i + j$$")

    text("## Transposition without copying")
    text("Suppose we want to transpose our matrix.")

    B = A.T  # @inspect B

    b_shape = B.shape  # @inspect b_shape
    b_strides = B.stride()  # @inspect b_strides

    image("images/tensor_strides.png", width=700)

    text("Notice that the transposed tensor has:")
    text("- Shape `(3, 2)`.")
    text("- Strides `(1, 3)`.")
    text("- The same underlying storage as the original tensor.")

    shares_storage = A.untyped_storage().data_ptr() == B.untyped_storage().data_ptr()  # @inspect shares_storage

    text("**No data was copied!** Only the tensor metadata changed.")

    text("## Contiguous and non-contiguous tensors")
    a_contiguous = A.is_contiguous()  # @inspect a_contiguous
    b_contiguous = B.is_contiguous()  # @inspect b_contiguous

    text("A tensor is contiguous when its elements follow the expected memory order.")
    text("Our transposed tensor is non-contiguous.")

    C = B.contiguous()  # @inspect C

    c_strides = C.stride()  # @inspect c_strides
    c_contiguous = C.is_contiguous()  # @inspect c_contiguous

    text("Calling `.contiguous()` creates a contiguous copy when necessary.")

    text("## Generalization to N dimensions")
    text("For a tensor with indices and strides:")
    text(r"$$\operatorname{offset}(i_0,\ldots,i_{n-1}) = o + \sum_{k=0}^{n-1} i_k s_k$$")
    text("Here, `o` is the storage offset.")
    text("This is the foundation of PyTorch's strided tensor representation.")

    text("## Takeaways")
    text("- Dense tensors generally rely on flat storage and metadata.")
    text("- Shape determines the logical dimensions.")
    text("- Strides determine how indices map to memory.")
    text("- Operations such as transpose can create views without copying data.")
    text("- Contiguity matters for efficient numerical computation.")


if __name__ == "__main__":
    main()
