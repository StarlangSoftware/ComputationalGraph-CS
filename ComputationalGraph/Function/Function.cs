using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    public interface Function
    {
        Tensor calculate(Tensor value);
        Tensor derivative(Tensor value, Tensor backward);
        ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased);
    }
}