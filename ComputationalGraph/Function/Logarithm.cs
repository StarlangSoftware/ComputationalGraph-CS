using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Logarithm : Function
    {
        /// <summary>
        /// Applies the natural logarithm function to each element of the input tensor.
        /// </summary>
        /// <param name="value">The tensor whose elements are to be transformed using the natural logarithm.</param>
        /// <returns>A new tensor containing the logarithmic values of the input tensor, with the same shape as the input.</returns>
        public Tensor calculate(Tensor value)
        {
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)value.GetData();

            foreach (double oldValue in oldValues)
            {
                values.Add(System.Math.Log(oldValue));
            }

            return new Tensor(values, value.GetShape());
        }

        /// <summary>
        /// Computes the derivative of the Logarithm function.
        /// </summary>
        /// <param name="value">output of the Logarithm(x).</param>
        /// <param name="backward">Backward tensor.</param>
        /// <returns>Gradient value of the corresponding node.</returns>
        public Tensor derivative(Tensor value, Tensor backward)
        {
            List<double> values = new List<double>();
            List<double> tensorValues = (List<double>)value.GetData();
            List<double> backwardValues = (List<double>)backward.GetData();

            for (int i = 0; i < tensorValues.Count; i++)
            {
                double val = tensorValues[i];
                double derivative = 1.0 / System.Math.Exp(val);
                double backwardValue = backwardValues[i];
                values.Add(derivative * backwardValue);
            }

            return new Tensor(values, value.GetShape());
        }

        public virtual ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}