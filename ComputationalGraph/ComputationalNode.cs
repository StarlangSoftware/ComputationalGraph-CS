using Math;

namespace ComputationalGraph;

public class ComputationalNode {
    
    private readonly FunctionType? _functionType;
    private char _op;
    private Matrix? _value;
    private Matrix? _backward;
    private readonly bool _isLearnable;

    public ComputationalNode(bool learnable, FunctionType functionType) {
        _isLearnable = learnable;
        _functionType = functionType;
        _backward = null;
        _value = null;
    }

    public ComputationalNode(bool learnable, char op) {
        _isLearnable = learnable;
        _op = op;
        _backward = null;
        _value = null;
        _functionType = null;
    }

    public ComputationalNode(Matrix value, char op) {
        _value = value;
        _isLearnable = true;
        _op = op;
        _backward = null;
        _functionType = null;
    }

    public FunctionType? GetFunctionType() {
        return _functionType;
    }

    public char GetOperator() {
        return _op;
    }

    public Matrix? GetValue() {
        return _value;
    }

    public void SetValue(Matrix? value) {
        _value = value;
    }

    public void UpdateValue() {
        _value?.Add(_backward);
    }
    public bool IsLearnable() {
        return _isLearnable;
    }

    public Matrix? GetBackward() {
        return _backward;
    }

    public void SetBackward(Matrix? backward) {
        _backward = backward;
    }
}