using Math;

namespace ComputationalGraph;

public interface IFunction {
    Matrix? Calculate(Matrix? matrix);
    Matrix? Derivative(Matrix? matrix);
}