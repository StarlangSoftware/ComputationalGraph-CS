using System;
using System.Collections.Generic;
using ComputationalGraph.Node;

namespace ComputationalGraph.Function
{
    [Serializable]
    public class TanhShrink : Tanh
    {
        public override ComputationalNode addEdge(List<ComputationalNode> inputNodes, bool isBiased)
        {
            ComputationalNode tanh = new FunctionNode(false, this);
            inputNodes[0].add(tanh);

            ComputationalNode negativeTanh = new FunctionNode(false, new Negation());
            tanh.add(negativeTanh);

            ComputationalNode tanhShrink = new ComputationalNode(false, isBiased);
            inputNodes[0].add(tanhShrink);
            negativeTanh.add(tanhShrink);

            return tanhShrink;
        }
    }
}