using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Power : Function
    {
        private readonly int _n;

        /**
         * <summary>Creates a power function with the given exponent.</summary>
         *
         * <param name="n">Exponent of the power function.</param>
         */
        public Power(int n)
        {
            _n = n;
        }

        /**
         * <summary>Creates a power function with the default exponent.</summary>
         */
        public Power()
        {
            _n = 2;
        }

        /**
         * <summary>Raises each element of the tensor to the specified power.</summary>
         *
         * <param name="value">Input tensor.</param>
         * <returns>A tensor whose elements are raised to the specified power.</returns>
         */
        public Tensor Calculate(Tensor value)
        {
            var values = new List<double>();
            var tensorValues = (List<double>)value.GetData();

            foreach (var val in tensorValues)
            {
                values.Add(System.Math.Pow(val, _n));
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Computes the derivative of the power function.</summary>
         *
         * <param name="value">Output of the power function.</param>
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
                var derivative = _n * System.Math.Pow(System.Math.Pow(val, 1.0 / _n), _n - 1);
                var backwardValue = backwardValues[i];
                values.Add(derivative * backwardValue);
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Adds a power function edge to the computational graph.</summary>
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