using System;
using System.Collections.Generic;
using ComputationalGraph.Node;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class MeanSquaredErrorLoss : Negation
    {
        public override ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode negatedY = new FunctionNode(false, this);
            inputNodes[0].add(negatedY);

            ComputationalNode yMinusNegatedY = new ComputationalNode();
            negatedY.add(yMinusNegatedY);
            inputNodes[1].add(yMinusNegatedY);

            ComputationalNode newNode = new FunctionNode(isBiased, new Power());
            yMinusNegatedY.add(newNode);

            return newNode;
        }
    }
}