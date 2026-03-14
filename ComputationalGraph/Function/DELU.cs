using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class DELU : Function
    {
        private readonly double a;
        private readonly double b;
        private readonly double xc;

        public DELU(double a, double b, double xc)
        {
            this.a = a;
            this.b = b;
            this.xc = xc;
        }

        public DELU()
        {
            this.a = 1.0;
            this.b = 2.0;
            this.xc = 1.25643;
        }

        /// <summary>
        /// Computes the DELU activation for the given value tensor.
        /// </summary>
        /// <param name="value">The tensor whose values are to be computed.</param>
        /// <returns>DELU(x).</returns>
        public Tensor calculate(Tensor value)
        {
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)value.GetData();

            foreach (double oldValue in oldValues)
            {
                if (oldValue > this.xc)
                {
                    values.Add(oldValue);
                }
                else
                {
                    values.Add((System.Math.Exp(this.a * oldValue) - 1) / this.b);
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /// <summary>
        /// Computes the derivative of the DELU activation function.
        /// </summary>
        /// <param name="value">output of the DELU(x).</param>
        /// <param name="backward">Backward tensor.</param>
        /// <returns>Gradient value of the corresponding node.</returns>
        public Tensor derivative(Tensor value, Tensor backward)
        {
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)value.GetData();
            List<double> backwardValues = (List<double>)backward.GetData();

            for (int i = 0; i < oldValues.Count; i++)
            {
                double backwardValue = backwardValues[i];
                double oldValue = oldValues[i];

                if (oldValue > this.xc)
                {
                    values.Add(backwardValue);
                }
                else
                {
                    values.Add(backwardValue * ((oldValue * this.b + 1) * (this.a / this.b)));
                }
            }

            return new Tensor(values, value.GetShape());
        }

        public ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}