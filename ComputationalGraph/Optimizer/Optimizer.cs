using System.Collections.Generic;
using ComputationalGraph.Node;
using Math;

namespace ComputationalGraph.Optimizer;

public abstract class Optimizer
{
    protected double LearningRate;
    private readonly double _etaDecrease;
    
    public Optimizer(double learningRate, double etaDecrease)
    {
        LearningRate = learningRate;
        _etaDecrease = etaDecrease;
    }

    /**
     * <summary>Updates the learning rate of the optimizer.</summary>
     */
    public void SetLearningRate(double learningRate)
    {
        LearningRate *= learningRate;
    }

    /**
     * <summary>Checks if broadcasting be applied to the corresponding node.</summary>
     * <param name="node"> The node to check.</param>
     * <returns> The index of the dimension where broadcasting is to be applied. -1 if broadcasting is not to be applied.</returns>
     */
    private int Broadcast(ComputationalNode node)
    {
        var v = node.GetValue().GetShape();
        var b = node.GetBackward().GetShape();
        var index = -1;
        for (var i = 0; i < v.Length; i++)
        {
            if (v[i] != b[i])
            {
                if (v[i] == 1)
                {
                    if (index != -1)
                    {
                        return -1;
                    }
                    index = i;
                }
            }
        }
        return index;
    }

    /**
     * <summary>Recursive helper function to update the values of learnable nodes.</summary>
     * <param name="visited"> A set of visited nodes.</param>
     * <param name="node"> The current node being processed.</param>
     * <param name="nodeMap"> A map of nodes to their children.</param>
     */
    private void UpdateRecursive(HashSet<ComputationalNode> visited, ComputationalNode node,
        Dictionary<ComputationalNode, List<ComputationalNode>> nodeMap)
    {
        visited.Add(node);
        if (node.IsLearnable())
        {
            var index = Broadcast(node);
            if (index != -1)
            {
                var v = 1;
                var b = 1;
                for (var i = node.GetValue().GetShape().Length - 1; i >= index; i--)
                {
                    v *= node.GetValue().GetShape()[i];
                    b *= node.GetValue().GetShape()[i];
                }

                var backwardValues = node.GetBackward().GetData();
                var values = new double[node.GetValue().GetData().Count];
                for (var i = 0; i < backwardValues.Count; i++)
                {
                    for (var j = i; j < i + b; j++)
                    {
                        values[((j - i) % v) + v * (j / b)] += backwardValues[j];
                    }

                    i += b - 1;
                }
                var list =  new List<double>();
                foreach (var d in values)
                {
                    list.Add(d);
                }
                node.SetBackward(new Tensor(list, node.GetValue().GetShape()));
            }

            this.SetGradients(node);
            node.UpdateValue();
        }

        if (nodeMap.ContainsKey(node))
        {
            foreach (var child in nodeMap[node])
            {
                if (!visited.Contains(child))
                {
                    UpdateRecursive(visited, child, nodeMap);
                }
            }
        }
    }
    
    /**
     * <summary>Sets the gradients (backward values) of the node.</summary>
     * <param name="node"> The node whose gradients are to be set.</param>
     */
    protected abstract void SetGradients(ComputationalNode node);

    /**
     * <summary>Updates the values of all learnable nodes in the graph.</summary>
     * <param name="nodeMap"> A map of nodes to their children.</param>
     */
    public void UpdateValues(Dictionary<ComputationalNode, List<ComputationalNode>> nodeMap)
    {
        var visited = new HashSet<ComputationalNode>();
        foreach (var node in nodeMap.Keys)
        {
            if (!visited.Contains(node))
            {
                UpdateRecursive(visited, node, nodeMap);
            }
        }
    }
}