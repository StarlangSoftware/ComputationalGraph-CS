using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Softmax : Function
    {
        public Tensor calculate(Tensor tensor)
        {
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)tensor.GetData();

            int lastDimensionSize = tensor.GetShape()[tensor.GetShape().Length - 1];
            double sum = 0.0;
            List<double> sumList = new List<double>();

            for (int i = 0; i < oldValues.Count; i++)
            {
                sum += System.Math.Exp(oldValues[i]);
                if ((i + 1) % lastDimensionSize == 0)
                {
                    sumList.Add(sum);
                    sum = 0.0;
                }
            }

            for (int i = 0; i < oldValues.Count; i++)
            {
                values.Add(System.Math.Exp(oldValues[i]) / sumList[i / lastDimensionSize]);
            }

            return new Tensor(values, tensor.GetShape());
        }

        public Tensor derivative(Tensor tensor, Tensor backward)
        {
            int lastDimensionSize = tensor.GetShape()[tensor.GetShape().Length - 1];

            List<double> values = new List<double>();
            List<double> oldValuesTensor = (List<double>)tensor.GetData();
            List<double> oldValuesBackward = (List<double>)backward.GetData();

            double total = 0.0;

            for (int i = 0; i < oldValuesTensor.Count; i++)
            {
                total += oldValuesTensor[i] * oldValuesBackward[i];

                if ((i + 1) % lastDimensionSize == 0)
                {
                    int startIndex = i / lastDimensionSize;
                    for (int j = 0; j < lastDimensionSize; j++)
                    {
                        values.Add(oldValuesBackward[startIndex * lastDimensionSize + j] - total);
                    }
                    total = 0.0;
                }
            }

            return tensor.HadamardProduct(new Tensor(values, tensor.GetShape()));
        }

        public ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode newNode = new FunctionNode(isBiased, this);
            inputNodes[0].add(newNode);
            return newNode;
        }
    }
}