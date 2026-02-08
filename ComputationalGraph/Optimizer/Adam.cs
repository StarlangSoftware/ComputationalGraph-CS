using System.Collections.Generic;
using ComputationalGraph.Node;
using Math;

namespace ComputationalGraph.Optimizer;

public class Adam : SGDMomentum
{
    private readonly Dictionary<ComputationalNode, double[]> _momentumMap;
    private readonly double _beta2;
    private readonly double _epsilon;
    private double _currentBeta1;
    private double _currentBeta2;
    
    public Adam(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon) : base(learningRate, etaDecrease, beta1)
    {
        _momentumMap = new Dictionary<ComputationalNode, double[]>();
        _beta2 = beta2;
        _epsilon = epsilon;
        _currentBeta1 = 1;
        _currentBeta2 = 1;
    }

    /**
     * <summary>Calculates the gradient updates using the Adam optimization algorithm.
     * This implementation follows a multi-pass approach:
     * 
     * <li><b>First Pass:</b> Calculates the weighted current gradients for both the first moment (momentum)
     * and the second moment (velocity/squared gradients).</li>
     * <li><b>Second Pass (Conditional):</b> If historical data exists, adds the decayed previous
     * momentum and velocity values to the current ones.</li>
     * <li><b>State Update:</b> Stores the raw calculated moments into the history maps.</li>
     * <li><b>Bias Correction:</b> Normalizes the moments by dividing them by <code>(1 - (beta)^t)</code>
     * to account for initialization bias.</li>
     * <li><b>Final Pass:</b> Computes the parameter update using the adaptive learning rate formula:
     * <code>(new_momentum / (sqrt(new_velocity) + epsilon)) * learningRate</code>.</li>
     * </summary>
     *
     * <param name="node"> The node whose gradients are to be set.</param>
     */
    protected List<double> Calculate(ComputationalNode node)
    {
        var backwardSize = node.GetBackward().GetData().Count;
        var newValuesMomentum = new List<double>(backwardSize);
        var newValuesVelocity = new List<double>(backwardSize);
        for (var i = 0; i < backwardSize; i++)
        {
            var backwardValue = node.GetBackward().GetData()[i];
            newValuesMomentum.Add((1 - Momentum) * backwardValue);
            newValuesVelocity.Add((1 - _beta2) * backwardValue * backwardValue);
        }

        if (_momentumMap.ContainsKey(node))
        {
            for (var i = 0; i < newValuesVelocity.Count; i++)
            {
                newValuesVelocity[i] += _beta2 * VelocityMap[node][i];
                newValuesMomentum[i] += Momentum * _momentumMap[node][i];
            }
        }

        var momentumValues = new double[backwardSize];
        var velocityValues = new double[backwardSize];
        for (var i = 0; i < backwardSize; i++)
        {
            momentumValues[i] = newValuesMomentum[i];
            velocityValues[i] = newValuesVelocity[i];
        }

        _momentumMap[node] = momentumValues;
        VelocityMap[node] = velocityValues;
        for (var i = 0; i < newValuesMomentum.Count; i++)
        {
            newValuesMomentum[i] /= (1 - _currentBeta1);
            newValuesVelocity[i] /= (1 - _currentBeta2);
        }

        var newValues = new List<double>(newValuesMomentum.Count);
        for (var i = 0; i < newValuesMomentum.Count; i++)
        {
            newValues.Add((newValuesMomentum[i] / (System.Math.Sqrt(newValuesVelocity[i]) + _epsilon)) * LearningRate);
        }
        return newValues;
    }

    /**
     * <summary>Sets the gradients for the given node using the Adam optimization algorithm.</summary>
     * <param name="node">The node whose gradients are to be set.</param>
     */
    protected override void SetGradients(ComputationalNode node)
    {
        node.SetBackward(new Tensor(Calculate(node), node.GetBackward().GetShape()));
    }

    /**
     * <summary>Updates the values of all learnable nodes and momentum values of the graph.</summary>
     * <param name="nodeMap"> A map of nodes to their children.</param>
     */
    public new void UpdateValues(Dictionary<ComputationalNode, List<ComputationalNode>> nodeMap)
    {
        _currentBeta1 *= Momentum;
        _currentBeta2 *= _beta2;
        base.UpdateValues(nodeMap);
    }
}