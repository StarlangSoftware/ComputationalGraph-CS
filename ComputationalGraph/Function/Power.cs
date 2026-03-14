using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Power : Function
    {
        private readonly int n;

        public Power(int n)
        {
            this.n = n;
        }

        public Power()
        {
            this.n = 2;
        }

        public Tensor calculate(Tensor value)
        {
            List<double> values = new List<double>();
            List<double> tensorValues = (List<double>)value.GetData();

            foreach (double val in tensorValues)
            {
                values.Add(System.Math.Pow(val, n));
            }

            return new Tensor(values, value.GetShape());
        }

        public Tensor derivative(Tensor value, Tensor backward)
        {
            List<double> values = new List<double>();
            List<double> tensorValues = (List<double>)value.GetData();
            List<double> backwardValues = (List<double>)backward.GetData();

            for (int i = 0; i < tensorValues.Count; i++)
            {
                double val = tensorValues[i];
                double derivative = n * System.Math.Pow(System.Math.Pow(val, 1.0 / n), n - 1);
                double backwardValue = backwardValues[i];
                values.Add(derivative * backwardValue);
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