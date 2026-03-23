using System;
using System.Collections.Generic;
using ComputationalGraph.Node;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class TanhShrink : Tanh
    {
        /**
         * <summary>Adds a TanhShrink function edge to the computational graph.</summary>
         *
         * <param name="inputNodes">Input nodes of the function.</param>
         * <param name="isBiased">Indicates whether the created node is biased.</param>
         * <returns>The created computational node.</returns>
         */
        public override ComputationalNode AddEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            var tanh = new FunctionNode(false, this);
            inputNodes[0].Add(tanh);

            var negativeTanh = new FunctionNode(false, new Negation());
            tanh.Add(negativeTanh);

            var tanhShrink = new ComputationalNode(false, isBiased);
            inputNodes[0].Add(tanhShrink);
            negativeTanh.Add(tanhShrink);

            return tanhShrink;
        }
    }
}