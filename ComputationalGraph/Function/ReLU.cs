using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class ReLU : Function
    {
        public Tensor calculate(Tensor value)
        {
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)value.GetData();

            foreach (double oldValue in oldValues)
            {
                values.Add(System.Math.Max(oldValue, 0));
            }

            return new Tensor(values, value.GetShape());
        }

        public Tensor derivative(Tensor value, Tensor backward)
        {
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)value.GetData();
            List<double> backwardValues = (List<double>)backward.GetData();

            for (int i = 0; i < oldValues.Count; i++)
            {
                double oldValue = oldValues[i];
                double backwardValue = backwardValues[i];

                if (oldValue > 0)
                {
                    values.Add(backwardValue);
                }
                else
                {
                    values.Add(0.0);
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