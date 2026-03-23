using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class ELU : Function
    {
        private readonly double _a;

        /**
         * <summary>Creates an ELU activation function with the given alpha value.</summary>
         *
         * <param name="a">Alpha parameter of the ELU activation.</param>
         */
        public ELU(double a)
        {
            _a = a;
        }

        /**
         * <summary>Creates an ELU activation function with the default alpha value.</summary>
         */
        public ELU()
        {
            _a = 1.0;
        }

        /**
         * <summary>Computes the ELU activation for the given tensor.</summary>
         *
         * <param name="value">The tensor whose values are to be computed.</param>
         * <returns>ELU(x).</returns>
         */
        public Tensor Calculate(Tensor value)
        {
            var values = new List<double>();
            var oldValues = (List<double>)value.GetData();

            foreach (var oldValue in oldValues)
            {
                if (oldValue < 0)
                {
                    values.Add(_a * (System.Math.Exp(oldValue) - 1));
                }
                else
                {
                    values.Add(oldValue);
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Computes the derivative of the ELU activation function.</summary>
         *
         * <param name="value">Output of the ELU(x).</param>
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

                if (oldValue < 0)
                {
                    values.Add((oldValue + _a) * backwardValue);
                }
                else
                {
                    values.Add(backwardValue);
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Adds an ELU function edge to the computational graph.</summary>
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