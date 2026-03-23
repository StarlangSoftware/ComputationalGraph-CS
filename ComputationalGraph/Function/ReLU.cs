using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class ReLU : Function
    {
        /**
         * <summary>Computes the ReLU activation for the given tensor.</summary>
         *
         * <param name="value">The tensor whose values are to be computed.</param>
         * <returns>ReLU(x).</returns>
         */
        public Tensor Calculate(Tensor value)
        {
            var values = new List<double>();
            var oldValues = (List<double>)value.GetData();

            foreach (var oldValue in oldValues)
            {
                values.Add(System.Math.Max(oldValue, 0));
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Computes the derivative of the ReLU activation function.</summary>
         *
         * <param name="value">Output of the ReLU(x).</param>
         * <param name="backward">Backward tensor.</param>
         * <returns>Gradient value of the corresponding node.</returns>
         */
        public Tensor Derivative(Tensor value, Tensor backward)
        {
            var values = new List<double>();
            var oldValues = (List<double>)value.GetData();
            var backwardValues = (List<double>)backward.GetData();

            for (var i = 0; i < oldValues.Count; i++)
            {
                var oldValue = oldValues[i];
                var backwardValue = backwardValues[i];

                if (oldValue > 0)
                {
                    values.Add(backwardValue);
                }
                else
                {
                    values.Add(0.0);
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Adds a ReLU function edge to the computational graph.</summary>
         *
         * <param name="inputNodes">Input nodes of the function.</param>
         * <param name="isBiased">Indicates whether the created node is biased.</param>
         * <returns>The created computational node.</returns>
         */
        public ComputationalNode AddEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            var newNode = new FunctionNode(isBiased, this);
            inputNodes[0].Add(newNode);
            return newNode;
        }
    }
}