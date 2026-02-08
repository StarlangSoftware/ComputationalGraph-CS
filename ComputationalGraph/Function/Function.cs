using Math;

namespace ComputationalGraph.Function;

public interface Function {
    Tensor Calculate(Tensor value);
    Tensor Derivative(Tensor value, Tensor backward);
}