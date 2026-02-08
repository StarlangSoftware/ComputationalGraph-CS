using ComputationalGraph.Node;
using Math;

namespace ComputationalGraph.Optimizer;

public class AdamW : Adam
{
    private readonly double _weightDecay;
    
    public AdamW(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon, double weightDecay) : base(learningRate, etaDecrease, beta1, beta2, epsilon)
    {
        _weightDecay = weightDecay;
    }

    /**
     * <summary>Sets the gradients for the given node using the AdamW optimization algorithm.</summary>
     * <param name="node"> The node whose gradients are to be set.</param>
     */
    protected override void SetGradients(ComputationalNode node)
    {
        var gradients = Calculate(node);
        var values = node.GetValue().GetData();
        for (var i = 0; i < gradients.Count; i++)
        {
            gradients[i] += LearningRate * _weightDecay * values[i];
        }
        node.SetBackward(new Tensor(gradients, node.GetBackward().GetShape()));
    }
}