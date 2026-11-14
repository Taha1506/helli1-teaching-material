from collections import defaultdict
from pathlib import Path

import torch
from torchviz import make_dot
from edtrace import text, image


def main():
    storage()
    tensor()
    computation_graph()


def storage():
    text("# PyTorch Internals: Storage and Autograd")
    text("## 1. Storage: Where Tensor Data Lives")
    text("A storage is a one-dimensional region of memory containing the underlying bytes of tensor data.")
    text("Storage itself does not specify a tensor's shape, strides, or logical indexing.")
    text("PyTorch has historically exposed two storage interfaces:")
    text("- `UntypedStorage` manages raw bytes without interpreting their datatype.")
    text("- `TypedStorage` associates storage with a datatype, but this interface is deprecated.")

    text("### Creating Raw Storage")
    text("`torch.UntypedStorage(40)` allocates 40 bytes of uninitialized storage; it does not create 40 floating-point elements.")
    untyped = torch.UntypedStorage(40)

    text("### The Deprecated TypedStorage Interface")
    text("A float32 element occupies four bytes, so this 40-byte allocation can hold ten float32 values.")
    text("The example below illustrates the historical API; use tensors for new code.")
    typed = torch.TypedStorage(wrap_storage=untyped, dtype=torch.float32, _internal=True)

    text("### Constructing a Tensor from Untyped Storage")
    text("We can attach a tensor's datatype, shape, stride, and storage offset to an existing storage allocation.")
    text("Here, 16 bytes are sufficient for a contiguous 2 × 2 float32 tensor.")
    untyped = torch.UntypedStorage(16)
    t = torch.empty(0, dtype=torch.float32)
    t.set_(source=untyped, storage_offset=0, size=(2, 2), stride=(2, 1))
    t.fill_(0.0)
    storage_tensor = t  # @inspect storage_tensor
    text("We initialize the tensor after attaching storage, because freshly allocated storage contains unspecified values.")


def tensor():
    text("## 2. Tensor Representation: Interpreting Storage")
    text("A tensor view is described by five key properties: storage, datatype, storage offset, shape, and strides.")
    text("- Storage identifies the underlying memory allocation.")
    text("- Datatype determines how the bytes are interpreted as elements.")
    text("- Storage offset specifies where the first tensor element begins, measured in elements.")
    text("- Shape gives the size along each dimension.")
    text("- Strides tell us how many elements to move in storage when advancing one index along each dimension.")

    a = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
    stride = a.stride()  # @inspect stride
    shape = a.size()  # @inspect shape
    offset = a.storage_offset()  # @inspect offset
    storage = a.untyped_storage()  # @inspect storage
    dtype = a.dtype  # @inspect dtype

    image("images/tensor_representation.png", width=500)

    text("### How Strides Translate Indices into Memory Locations")
    text("For a two-dimensional tensor, the element at `(i, j)` is found at `storage_offset + i * stride[0] + j * stride[1]`, in units of elements.")
    text("In this example, the shape is `(2, 3)` and the strides are `(3, 1)` because rows are stored contiguously.")

    text("### Creating a View with Explicit Strides")
    text("`torch.as_strided` constructs a view of an existing tensor without copying its underlying storage.")
    base = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    strided = torch.as_strided(input=base, storage_offset=0, size=(2, 2), stride=(2, 1))  # @inspect strided
    text("The view and its base share storage. Changing a value through either tensor can affect the other.")
    text("Strided views must remain within the allocated storage; overlapping views also require care when writing.")


def computation_graph():
    text("## 3. PyTorch Autograd: The Computation Graph")
    text("When an operation on tensors is tracked by autograd, PyTorch dynamically records the information needed to differentiate its result.")
    text("The autograd graph consists primarily of backward-function nodes and edges connecting those functions.")
    text("Each backward node implements a local vector-Jacobian product, which propagates an incoming gradient toward its inputs.")
    text("Unlike a conventional forward computation diagram, `grad_fn` and `next_functions` expose the backward graph.")

    text("### Leaf Tensors and AccumulateGrad")
    text("A leaf tensor with `requires_grad=True` has an associated `AccumulateGrad` node, which adds incoming gradients to its `.grad` field.")
    text("A leaf tensor's `grad_fn` is `None`, but we can access its gradient edge with `torch.autograd.graph.get_gradient_edge()`.")

    a = torch.tensor(3.0, requires_grad=True)
    gradient_edge = torch.autograd.graph.get_gradient_edge(a)
    gradient_node = gradient_edge.node
    gradient_node_type = type(gradient_node)  # @inspect gradient_node_type

    text("### Backward Nodes for Operations")
    text("Non-leaf tensors produced by differentiable operations usually have a `grad_fn` pointing to the operation's backward node.")

    b = a ** 2
    gradient_edge = torch.autograd.graph.get_gradient_edge(b)
    gradient_node = gradient_edge.node
    gradient_node_type = type(gradient_node)  # @inspect gradient_node_type

    text("For `b = a ** 2`, the backward node is normally called `PowBackward0`.")
    text("A backward node can be called with an incoming gradient to compute gradients for the operation's inputs.")

    local_gradient = gradient_node(torch.tensor(1.0))  # @inspect local_gradient

    text("Here, `a = 3`, so the derivative of `a ** 2` is `2 * a = 6`. An incoming gradient of 1 produces a gradient of 6.")
    text("Calling the operation's backward node directly does not traverse its predecessors or populate the leaf tensor's `.grad`.")

    text("### Operations with Multiple Outputs")
    text("One forward operation can return several tensors that share a single backward node.")
    text("A `GradientEdge` identifies both the backward node and the output slot, `output_nr`, associated with a tensor.")

    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    x, y, z = torch.chunk(a, 3)

    gradient_edge_x = torch.autograd.graph.get_gradient_edge(x)
    gradient_edge_y = torch.autograd.graph.get_gradient_edge(y)
    gradient_edge_z = torch.autograd.graph.get_gradient_edge(z)

    output_nr_x = gradient_edge_x.output_nr  # @inspect output_nr_x
    output_nr_y = gradient_edge_y.output_nr  # @inspect output_nr_y
    output_nr_z = gradient_edge_z.output_nr  # @inspect output_nr_z

    same_node = gradient_edge_x.node is gradient_edge_y.node and gradient_edge_y.node is gradient_edge_z.node  # @inspect same_node

    text("All three chunks share one `SplitBackward0` node, but their output numbers are 0, 1, and 2.")

    text("### Traversing the Backward Graph")
    text("A node's `next_functions` contains pairs of `(previous_node, output_nr)` describing its connections toward the graph's inputs.")
    text("The output number in each pair identifies the slot of the preceding backward node to which a gradient must be sent.")

    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = a ** 2
    c = b.sum()

    next_functions = c.grad_fn.next_functions  # @inspect next_functions

    text("The backward path here is `SumBackward0 → PowBackward0 → AccumulateGrad`.")

    text("### Implementing a Simplified Backward Engine")
    text("We can simulate backpropagation by calling backward nodes and following `next_functions` ourselves.")
    text("A shared node may receive gradients from several branches, so we must add those contributions before executing it.")
    text("We first find a topological order, then execute nodes backward while collecting gradients by node and output number.")

    a = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = a ** 2
    c = b.sum()

    manual_backward(c)
    manual_grad = a.grad.clone()  # @inspect manual_grad

    text("Because `c = a[0] ** 2 + a[1] ** 2 + a[2] ** 2`, the gradient with respect to `a` is `[2, 4, 6]`.")

    text("### A Graph with Six Branches")
    text("Now we split a tensor into six outputs and add those outputs together.")
    text("All six branches share the same splitting node, so their gradients must converge before that node runs.")

    a = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], requires_grad=True)
    b1, b2, b3, b4, b5, b6 = torch.chunk(a, chunks=6)
    c = b1 + b2 + b3 + b4 + b5 + b6

    manual_backward(c, torch.ones_like(c))
    branch_gradient = a.grad.clone()  # @inspect branch_gradient

    text("Each element of `a` contributes to the output exactly once, giving the gradient `[1, 1, 1, 1, 1, 1]`.")

    text("### Visualizing the Graph with Torchviz")
    text("`torchviz.make_dot` displays backward functions, leaf tensors, and the specified output tensor.")
    text("The `params` argument labels leaf tensors such as `a`; to label the output `c`, we explicitly update its Graphviz node.")

    Path("images").mkdir(parents=True, exist_ok=True)

    graph = make_dot(c, params={"a": a})
    graph.node(str(id(c)), "c\n(1)", fillcolor="darkolivegreen1")
    graph.render(filename="computation_graph", directory="images", format="png", cleanup=True)

    image("images/computation_graph.png")

    text("The six paths begin at one `SplitBackward0` node and meet through a chain of addition nodes.")
    text("This makes both gradient accumulation and the distinction between backward nodes and output tensors visible.")


def manual_backward(output, gradient=None):
    """Educational backward engine supporting branching and multi-output nodes."""

    if gradient is None:
        if output.numel() != 1:
            raise ValueError("A non-scalar output needs an explicit gradient.")
        gradient = torch.ones_like(output)

    root = torch.autograd.graph.get_gradient_edge(output)
    visited = set()
    postorder = []

    def visit(node):
        if node is None or node in visited:
            return

        visited.add(node) # @inspect visited

        for previous_node, _ in node.next_functions:
            visit(previous_node)

        postorder.append(node) # @inspect postorder

    visit(root.node)

    gradients = defaultdict(dict)
    gradients[root.node][root.output_nr] = gradient

    with torch.no_grad():
        for node in reversed(postorder):
            output_grads = gradients.get(node, {})

            if not output_grads:
                continue

            args = tuple(output_grads.get(i) for i in range(max(output_grads) + 1))
            input_grads = node(*args)

            if not isinstance(input_grads, tuple):
                input_grads = (input_grads,)

            for (previous_node, output_nr), grad in zip(node.next_functions, input_grads):
                if previous_node is None or grad is None:
                    continue

                slot = gradients[previous_node]
                slot[output_nr] = slot[output_nr] + grad if output_nr in slot else grad # @inspect gradients


if __name__ == "__main__":
    main()
