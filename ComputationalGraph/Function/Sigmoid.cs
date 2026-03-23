using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Sigmoid : Function
    {
        /**
         * <summary>Computes the sigmoid activation for the given tensor.</summary>
         *
         * <param name="value">The tensor whose values are to be computed.</param>
         * <returns>Sigmoid(x).</returns>
         */
        public Tensor Calculate(Tensor value)
        {
            var values = new List<double>();
            var tensorValues = (List<double>)value.GetData();

            foreach (var val in tensorValues)
            {
                var sigmoid = 1.0 / (1.0 + System.Math.Exp(-val));
                values.Add(sigmoid);
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Computes the derivative of the sigmoid activation function.</summary>
         *
         * <param name="value">Output of the sigmoid function.</param>
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
                var derivative = val * (1 - val);
                var backwardValue = backwardValues[i];
                values.Add(derivative * backwardValue);
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Adds a sigmoid function edge to the computational graph.</summary>
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