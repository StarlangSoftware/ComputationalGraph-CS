using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    /**
     * <summary>Defines the behavior of functions used in the computational graph.</summary>
     */
    public interface Function
    {
        /**
         * <summary>Calculates the output tensor for the given input tensor.</summary>
         *
         * <param name="value">Input tensor.</param>
         * <returns>The calculated output tensor.</returns>
         */
        Tensor Calculate(Tensor value);

        /**
         * <summary>Calculates the derivative of the function using the given value and backward tensor.</summary>
         *
         * <param name="value">Input tensor value.</param>
         * <param name="backward">Backward tensor.</param>
         * <returns>The derivative tensor.</returns>
         */
        Tensor Derivative(Tensor value, Tensor backward);

        /**
         * <summary>Adds a function edge to the computational graph.</summary>
         *
         * <param name="inputNodes">Input nodes of the function.</param>
         * <param name="isBiased">Indicates whether the created node is biased.</param>
         * <returns>The created computational node.</returns>
         */
        ComputationalNode AddEdge(List<ComputationalNode> inputNodes, bool isBiased);
    }
}