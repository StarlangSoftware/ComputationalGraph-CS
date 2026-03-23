using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class Softmax : Function
    {
        /**
         * <summary>Computes the softmax activation for the given tensor.</summary>
         *
         * <param name="tensor">The tensor whose values are to be normalized.</param>
         * <returns>Softmax output tensor.</returns>
         */
        public Tensor Calculate(Tensor tensor)
        {
            var values = new List<double>();
            var oldValues = (List<double>)tensor.GetData();

            var lastDimensionSize = tensor.GetShape()[tensor.GetShape().Length - 1];
            var sum = 0.0;
            var sumList = new List<double>();

            for (var i = 0; i < oldValues.Count; i++)
            {
                sum += System.Math.Exp(oldValues[i]);
                if ((i + 1) % lastDimensionSize == 0)
                {
                    sumList.Add(sum);
                    sum = 0.0;
                }
            }

            for (var i = 0; i < oldValues.Count; i++)
            {
                values.Add(System.Math.Exp(oldValues[i]) / sumList[i / lastDimensionSize]);
            }

            return new Tensor(values, tensor.GetShape());
        }

        /**
         * <summary>Computes the derivative of the softmax function.</summary>
         *
         * <param name="tensor">Output of the softmax function.</param>
         * <param name="backward">Backward tensor.</param>
         * <returns>Gradient value of the corresponding node.</returns>
         */
        public Tensor Derivative(Tensor tensor, Tensor backward)
        {
            var lastDimensionSize = tensor.GetShape()[tensor.GetShape().Length - 1];

            var values = new List<double>();
            var oldValuesTensor = (List<double>)tensor.GetData();
            var oldValuesBackward = (List<double>)backward.GetData();

            var total = 0.0;

            for (var i = 0; i < oldValuesTensor.Count; i++)
            {
                total += oldValuesTensor[i] * oldValuesBackward[i];

                if ((i + 1) % lastDimensionSize == 0)
                {
                    var startIndex = i / lastDimensionSize;
                    for (var j = 0; j < lastDimensionSize; j++)
                    {
                        values.Add(oldValuesBackward[startIndex * lastDimensionSize + j] - total);
                    }

                    total = 0.0;
                }
            }

            return tensor.HadamardProduct(new Tensor(values, tensor.GetShape()));
        }

        /**
         * <summary>Adds a softmax function edge to the computational graph.</summary>
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