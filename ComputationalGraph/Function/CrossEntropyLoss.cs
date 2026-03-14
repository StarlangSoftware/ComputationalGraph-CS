using System;
using System.Collections.Generic;
using ComputationalGraph.Node;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class CrossEntropyLoss : Logarithm
    {
        public override ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode logy = new FunctionNode(false, this);
            inputNodes[0].add(logy);

            ComputationalNode ylogy = new MultiplicationNode(false, isBiased, true);
            inputNodes[1].add(ylogy);
            logy.add(ylogy);

            return ylogy;
        }
    }
}