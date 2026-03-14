using System;
using System.Collections.Generic;
using ComputationalGraph.Node;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class SiLU : Sigmoid
    {
        public override ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode sigmoid = new FunctionNode(false, this);
            inputNodes[0].add(sigmoid);

            ComputationalNode swish = new MultiplicationNode(false, isBiased, true);
            sigmoid.add(swish);
            inputNodes[0].add(swish);

            return swish;
        }
    }
}