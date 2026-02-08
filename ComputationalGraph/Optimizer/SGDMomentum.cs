using System.Collections.Generic;
using ComputationalGraph.Node;
using Math;

namespace ComputationalGraph.Optimizer;

public class SGDMomentum : Optimizer
{
    protected readonly double Momentum;
    protected readonly Dictionary<ComputationalNode, double[]> VelocityMap;
    
    public SGDMomentum(double learningRate, double etaDecrease, double momentum) : base(learningRate, etaDecrease)
    {
        Momentum = momentum;
        VelocityMap = new Dictionary<ComputationalNode, double[]>();
    }

    /**
     * <summary>Calculates the new gradients by combining the current gradient with the previous velocity.
     * It updates the internal velocity state and modifies the node's backward tensor
     * to reflect the momentum-adjusted update step.</summary>
     *
     * <param name="node"> The node whose gradients are to be set.</param>
     */
    protected override void SetGradients(ComputationalNode node)
    {
        var backwardSize = node.GetBackward().GetData().Count;
        var newValues = new List<double>(backwardSize);
        for (var i = 0; i < backwardSize; i++)
        {
            newValues.Add((1 - Momentum) * node.GetBackward().GetData()[i]);
        }

        if (VelocityMap.ContainsKey(node))
        {
            for (var i = 0; i < newValues.Count; i++)
            {
                newValues[i] += VelocityMap[node][i] * Momentum;
            }
        }
        var velocity = new double[backwardSize];
        for (var i = 0; i < backwardSize; i++)
        {
            velocity[i] = newValues[i];
        }
        VelocityMap[node] = velocity;
        for (var i = 0; i < newValues.Count; i++)
        {
            newValues[i] *= LearningRate;
        }
        node.SetBackward(new Tensor(newValues, node.GetBackward().GetShape()));
    }
}