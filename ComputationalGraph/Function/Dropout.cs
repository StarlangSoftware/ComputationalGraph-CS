using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Dropout : Function
    {
        private readonly double p;
        private readonly Random random;
        private readonly List<double> mask;

        public Dropout(double p, Random random)
        {
            this.p = p;
            this.random = random;
            this.mask = new List<double>();
        }

        /// <summary>
        /// Computes the dropout values for the given value tensor.
        /// </summary>
        /// <param name="value">The tensor whose values are to be computed.</param>
        /// <returns>Output tensor.</returns>
        public Tensor calculate(Tensor value)
        {
            this.mask.Clear();
            double multiplier = 1.0 / (1 - p);

            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)value.GetData();

            foreach (double oldValue in oldValues)
            {
                double r = random.NextDouble();
                if (r > p)
                {
                    mask.Add(multiplier);
                    values.Add(oldValue * multiplier);
                }
                else
                {
                    mask.Add(0.0);
                    values.Add(0.0);
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /// <summary>
        /// Calculates the derivative of the dropout.
        /// </summary>
        /// <param name="value">output of the dropout function.</param>
        /// <param name="backward">Backward tensor.</param>
        /// <returns>Gradient value of the corresponding node.</returns>
        public Tensor derivative(Tensor value, Tensor backward)
        {
            return backward.HadamardProduct(new Tensor(mask, value.GetShape()));
        }

        public ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}