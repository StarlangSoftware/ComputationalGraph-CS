using System;
using System.Collections.Generic;
using ComputationalGraph.Node;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class MeanSquaredErrorLoss : Negation
    {
        /**
         * <summary>Adds a mean squared error loss edge to the computational graph.</summary>
         *
         * <param name="inputNodes">Input nodes of the loss function.</param>
         * <param name="isBiased">Indicates whether the created node is biased.</param>
         * <returns>The created computational node.</returns>
         */
        public override ComputationalNode AddEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            var negatedY = new FunctionNode(false, this);
            inputNodes[0].Add(negatedY);

            var yMinusNegatedY = new ComputationalNode();
            negatedY.Add(yMinusNegatedY);
            inputNodes[1].Add(yMinusNegatedY);

            var newNode = new FunctionNode(isBiased, new Power());
            yMinusNegatedY.Add(newNode);

            return newNode;
        }
    }
}