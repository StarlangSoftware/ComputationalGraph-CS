using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Logarithm : Function
    {
        /**
         * <summary>Applies the natural logarithm function to each element of the input tensor.</summary>
         *
         * <param name="value">The tensor whose elements are to be transformed using the natural logarithm.</param>
         * <returns>A new tensor containing the logarithmic values of the input tensor.</returns>
         */
        public Tensor Calculate(Tensor value)
        {
            var values = new List<double>();
            var oldValues = (List<double>)value.GetData();

            foreach (var oldValue in oldValues)
            {
                values.Add(System.Math.Log(oldValue));
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Computes the derivative of the logarithm function.</summary>
         *
         * <param name="value">Output of the logarithm function.</param>
         * <param name="backward">Backward tensor.</param>
         * <returns>Gradient value of the corresponding node.</returns>
         */
        public Tensor Derivative(Tensor value, Tensor backward)
        {
            var values = new List<double>();
            var tensorValues = (List<double>)value.GetData();
            var backwardValues = (List<double>)backward.GetData();

            for (var i = 0; i < tensorValues.Count; i++)
            {
                var val = tensorValues[i];
                var derivative = 1.0 / System.Math.Exp(val);
                var backwardValue = backwardValues[i];
                values.Add(derivative * backwardValue);
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Adds a logarithm function edge to the computational graph.</summary>
         *
         * <param name="inputNodes">Input nodes of the function.</param>
         * <param name="isBiased">Indicates whether the created node is biased.</param>
         * <returns>The created computational node.</returns>
         */
        public virtual ComputationalNode AddEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            var newNode = new FunctionNode(isBiased, this);
            inputNodes[0].Add(newNode);
            return newNode;
        }
    }
}