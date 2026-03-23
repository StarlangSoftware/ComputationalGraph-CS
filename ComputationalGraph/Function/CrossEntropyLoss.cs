using System;
using System.Collections.Generic;
using ComputationalGraph.Node;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class CrossEntropyLoss : Logarithm
    {
        /**
         * <summary>Adds a cross-entropy loss edge to the computational graph.</summary>
         *
         * <param name="inputNodes">Input nodes of the loss function.</param>
         * <param name="isBiased">Indicates whether the created node is biased.</param>
         * <returns>The created computational node.</returns>
         */
        public override ComputationalNode AddEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            var logy = new FunctionNode(false, this);
            inputNodes[0].Add(logy);

            var ylogy = new MultiplicationNode(false, isBiased, true);
            inputNodes[1].Add(ylogy);
            logy.Add(ylogy);

            return ylogy;
        }
    }
}