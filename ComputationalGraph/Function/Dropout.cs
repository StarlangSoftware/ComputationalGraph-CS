using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Dropout : Function
    {
        private readonly double _p;
        private readonly Random _random;
        private readonly List<double> _mask;

        /**
         * <summary>Creates a dropout function with the given dropout probability and random generator.</summary>
         *
         * <param name="p">Dropout probability.</param>
         * <param name="random">Random object used for generating the dropout mask.</param>
         */
        public Dropout(double p, Random random)
        {
            _p = p;
            _random = random;
            _mask = new List<double>();
        }

        /**
         * <summary>Computes the dropout values for the given value tensor.</summary>
         *
         * <param name="value">The tensor whose values are to be computed.</param>
         * <returns>Output tensor.</returns>
         */
        public Tensor Calculate(Tensor value)
        {
            _mask.Clear();
            var multiplier = 1.0 / (1 - _p);

            var values = new List<double>();
            var oldValues = (List<double>)value.GetData();

            foreach (var oldValue in oldValues)
            {
                var randomValue = _random.NextDouble();
                if (randomValue > _p)
                {
                    _mask.Add(multiplier);
                    values.Add(oldValue * multiplier);
                }
                else
                {
                    _mask.Add(0.0);
                    values.Add(0.0);
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /**
         * <summary>Calculates the derivative of the dropout function.</summary>
         *
         * <param name="value">Output of the dropout function.</param>
         * <param name="backward">Backward tensor.</param>
         * <returns>Gradient value of the corresponding node.</returns>
         */
        public Tensor Derivative(Tensor value, Tensor backward)
        {
            return backward.HadamardProduct(new Tensor(_mask, value.GetShape()));
        }

        /**
         * <summary>Adds a dropout function edge to the computational graph.</summary>
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