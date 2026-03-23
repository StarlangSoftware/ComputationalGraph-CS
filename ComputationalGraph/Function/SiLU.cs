using System;
using System.Collections.Generic;
using ComputationalGraph.Node;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class SiLU : Sigmoid
    {
        /**
         * <summary>Adds a SiLU function edge to the computational graph.</summary>
         *
         * <param name="inputNodes">Input nodes of the function.</param>
         * <param name="isBiased">Indicates whether the created node is biased.</param>
         * <returns>The created computational node.</returns>
         */
        public override ComputationalNode AddEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            var sigmoid = new FunctionNode(false, this);
            inputNodes[0].Add(sigmoid);

            var swish = new MultiplicationNode(false, isBiased, true);
            sigmoid.Add(swish);
            inputNodes[0].Add(swish);

            return swish;
        }
    }
}