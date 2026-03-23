using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Negation : Function
    {
        /**
         * <summary>Negates the values of the given tensor.</summary>
         *
         * <param name="value">The tensor whose values are to be negated.</param>
         * <returns>The negated tensor.</returns>
         */
        public Tensor Calculate(Tensor value)
        {
            var values = new List<double>();
            var oldValues = (List<double>)value.GetData();

            foreach (var oldValue in oldValues)
            {
                values.Add(-oldValue);
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Calculates the derivative of the negation function.</summary>
         *
         * <param name="value">Output of the negation function.</param>
         * <param name="backward">Backward tensor.</param>
         * <returns>Gradient value of the corresponding node.</returns>
         */
        public Tensor Derivative(Tensor value, Tensor backward)
        {
            var values = new List<double>();
            var backwardValues = (List<double>)backward.GetData();

            foreach (var backwardValue in backwardValues)
            {
                values.Add(-backwardValue);
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Adds a negation function edge to the computational graph.</summary>
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