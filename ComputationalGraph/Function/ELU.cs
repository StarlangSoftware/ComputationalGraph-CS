using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class ELU : Function
    {
        private readonly double a;

        public ELU(double a)
        {
            this.a = a;
        }

        public ELU()
        {
            this.a = 1.0;
        }

        /// <summary>
        /// Computes the ELU activation for the given tensor.
        /// </summary>
        /// <param name="value">The tensor whose values are to be computed.</param>
        /// <returns>ELU(x).</returns>
        public Tensor calculate(Tensor value)
        {
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)value.GetData();

            foreach (double oldValue in oldValues)
            {
                if (oldValue < 0)
                {
                    values.Add(a * (System.Math.Exp(oldValue) - 1));
                }
                else
                {
                    values.Add(oldValue);
                }
            }

            return new Tensor(values, value.GetShape());
        }

        /// <summary>
        /// Computes the derivative of the ELU activation function.
        /// </summary>
        /// <param name="value">output of the ELU(x).</param>
        /// <param name="backward">Backward tensor.</param>
        /// <returns>Gradient value of the corresponding node.</returns>
        public Tensor derivative(Tensor value, Tensor backward)
        {
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)value.GetData();
            List<double> backwardValues = (List<double>)backward.GetData();

            for (int i = 0; i < oldValues.Count; i++)
            {
                double oldValue = oldValues[i];
                double backwardValue = backwardValues[i];

                if (oldValue < 0)
                {
                    values.Add((oldValue + a) * backwardValue);
                }
                else
                {
                    values.Add(backwardValue);
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