using System.Collections.Generic;
using ComputationalGraph.Node;
using Math;

namespace ComputationalGraph.Optimizer;

public class StochasticGradientDescent : Optimizer
{
    public StochasticGradientDescent(double learningRate, double etaDecrease) : base(learningRate, etaDecrease)
    {
    }

    /**
     * <summary>Sets the gradients (backward values) of the node to the learning rate times the backward values.</summary>
     * <param name="node">The node whose gradients are to be set.</param>
     */
    protected override void SetGradients(ComputationalNode node)
    {
        var values = new List<double>();
        var backward = node.GetBackward().GetData();
        foreach (var aDouble in backward)
        {
            values.Add(aDouble * LearningRate);
        }
        node.SetBackward(new Tensor(values, node.GetBackward().GetShape()));
    }
}