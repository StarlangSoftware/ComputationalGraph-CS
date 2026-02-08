using Math;

namespace ComputationalGraph.Node;

public class MultiplicationNode : ComputationalNode
{
    private readonly bool _isHadamard;
    private readonly ComputationalNode _priorityNode;
    
    public MultiplicationNode(bool learnable, bool isBiased, bool isHadamard) : base(learnable, isBiased)
    {
        _isHadamard = isHadamard;
        _priorityNode = null;
    }

    public MultiplicationNode(bool learnable, bool isBiased, bool isHadamard, ComputationalNode priorityNode) : base(
        learnable, isBiased)
    {
        _isHadamard = isHadamard;
        _priorityNode = priorityNode;
    }
    
    public MultiplicationNode(bool learnable, bool isBiased, Tensor value, bool isHadamard) : base(learnable, isBiased)
    {
        Value = value;
        _isHadamard = isHadamard;
        _priorityNode = null;
    }

    public MultiplicationNode(bool learnable, Tensor value) : base(learnable, false)
    {
        Value = value;
        _isHadamard = false;
        _priorityNode = null;
    }

    public MultiplicationNode(Tensor value) : base(true, false)
    {
        Value = value;
        _isHadamard = false;
        _priorityNode = null;
    }
    
    public MultiplicationNode(bool learnable, bool isBiased) : base(learnable, isBiased)
    {
        _isHadamard = false;
        _priorityNode = null;
    }

    public bool IsHadamard()
    {
        return _isHadamard;
    }

    public ComputationalNode GetPriorityNode()
    {
        return _priorityNode;
    }

}