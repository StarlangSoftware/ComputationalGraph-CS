using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class DELU : Function
    {
        private readonly double _a;
        private readonly double _b;
        private readonly double _xc;

        /**
         * <summary>Creates a DELU activation function with the given parameters.</summary>
         *
         * <param name="a">Scaling parameter used in the exponential part.</param>
         * <param name="b">Normalization parameter used in the exponential part.</param>
         * <param name="xc">Threshold value for switching to the linear region.</param>
         */
        public DELU(double a, double b, double xc)
        {
            _a = a;
            _b = b;
            _xc = xc;
        }

        /**
         * <summary>Creates a DELU activation function with default parameters.</summary>
         */
        public DELU()
        {
            _a = 1.0;
            _b = 2.0;
            _xc = 1.25643;
        }

        /**
         * <summary>Computes the DELU activation for the given value tensor.</summary>
         *
         * <param name="value">The tensor whose values are to be computed.</param>
         * <returns>DELU(x).</returns>
         */
        public Tensor Calculate(Tensor value)
        {
            var values = new List<double>();
            var oldValues = (List<double>)value.GetData();

            foreach (var oldValue in oldValues)
            {
                if (oldValue > _xc)
                {
                    values.Add(oldValue);
                }
                else
                {
                    values.Add((System.Math.Exp(_a * oldValue) - 1) / _b);
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Computes the derivative of the DELU activation function.</summary>
         *
         * <param name="value">Output of the DELU(x).</param>
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
                var backwardValue = backwardValues[i];
                var oldValue = oldValues[i];

                if (oldValue > _xc)
                {
                    values.Add(backwardValue);
                }
                else
                {
                    values.Add(backwardValue * ((oldValue * _b + 1) * (_a / _b)));
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Adds a DELU function edge to the computational graph.</summary>
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