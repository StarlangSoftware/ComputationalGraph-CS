using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Sigmoid : Function
    {
        public Tensor calculate(Tensor value)
        {
            List<double> values = new List<double>();
            List<double> tensorValues = (List<double>)value.GetData();

            foreach (double val in tensorValues)
            {
                double sigmoid = 1.0 / (1.0 + System.Math.Exp(-val));
                values.Add(sigmoid);
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
                double derivative = val * (1 - val);
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