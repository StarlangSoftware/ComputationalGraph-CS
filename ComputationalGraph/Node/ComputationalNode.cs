using Math;

namespace ComputationalGraph.Node;

public class ComputationalNode {
    
    protected Tensor Value;
    protected Tensor Backward;
    protected readonly bool _isLearnable;
    protected readonly bool _isBiased;
    protected readonly Function.Function Function;

    /**
     * <summary>Initializes a ComputationalNode.</summary>
     * <param name="learnable"> Indicates whether the node is learnable (e.g., weights)</param>
     * <param name="isBiased"> Indicates whether the node is biased</param>
     * <param name="function"> The function (e.g., activation like SIGMOID)</param>
     * <param name="value"> The tensor value associated with the node (optional)</param>
     */
    public ComputationalNode(bool learnable, bool isBiased, Function.Function function, Tensor value) {
        _isLearnable = learnable;
        Backward = null;
        Value = value;
        _isBiased = isBiased;
        Function = function;
    }

    /**
     * Constructor overload for function type initialization
     * <param name="learnable"> Indicates whether the node is learnable (e.g., weights)</param>
     * <param name="isBiased"> Indicates whether the node is biased</param>
     * <param name="function"> The function (e.g., activation like SIGMOID)</param>
     */
    public ComputationalNode(bool learnable, Function.Function function, bool isBiased) : this(learnable, isBiased, function, null) {
    }

    public ComputationalNode(bool learnable, bool isBiased) : this(learnable, isBiased, null, null) {
    }

    public bool IsBiased()
    {
        return _isBiased;
    }

    public Function.Function GetFunction()
    {
        return Function;
    }
    
    public Tensor GetValue() {
        return Value;
    }

    public void SetValue(Tensor value) {
        Value = value;
    }

    public void UpdateValue() {
        Value.Add(Backward);
    }
    
    public bool IsLearnable() {
        return _isLearnable;
    }

    public Tensor GetBackward() {
        return Backward;
    }

    public void SetBackward(Tensor backward) {
        Backward = backward;
    }
}