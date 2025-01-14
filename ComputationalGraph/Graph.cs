using Math;

namespace ComputationalGraph;

public class Graph {
    
    private readonly Dictionary<ComputationalNode, List<ComputationalNode>> _nodeMap;
    private readonly Dictionary<ComputationalNode, List<ComputationalNode>> _reverseNodeMap;

    public Graph() {
        _nodeMap = new Dictionary<ComputationalNode, List<ComputationalNode>>();
        _reverseNodeMap = new Dictionary<ComputationalNode, List<ComputationalNode>>();
    }

    public ComputationalNode AddEdge(ComputationalNode first, ComputationalNode second) {
        var newNode = new ComputationalNode(false, second.GetOperator());
        if (!_nodeMap.ContainsKey(first)) {
            _nodeMap[first] = new List<ComputationalNode>();
        }
        if (!_nodeMap.ContainsKey(second)) {
            _nodeMap[second] = new List<ComputationalNode>();
        }
        _nodeMap[first].Add(newNode);
        _nodeMap[second].Add(newNode);
        if (!_reverseNodeMap.ContainsKey(newNode)) {
            _reverseNodeMap[newNode] = new List<ComputationalNode>();
        }
        _reverseNodeMap[newNode].Add(first);
        _reverseNodeMap[newNode].Add(second);
        return newNode;
    }

    public ComputationalNode AddEdge(ComputationalNode node, FunctionType type) {
        var newNode = new ComputationalNode(false, type);
        if (!_nodeMap.ContainsKey(node)) {
            _nodeMap[node] = new List<ComputationalNode>();
        }
        _nodeMap[node].Add(newNode);
        if (!_reverseNodeMap.ContainsKey(newNode)) {
            _reverseNodeMap[newNode] = new List<ComputationalNode>();
        }
        _reverseNodeMap[newNode].Add(node);
        return newNode;
    }

    private List<ComputationalNode> Sort(ComputationalNode root, HashSet<ComputationalNode> visited) {
        var queue = new List<ComputationalNode>();
        visited.Add(root);
        if (_nodeMap.ContainsKey(root)) {
            for (var i = 0; i < _nodeMap[root].Count; i++) {
                if (!visited.Contains(_nodeMap[root][i])) {
                    queue.AddRange(Sort(_nodeMap[root][i], visited));
                }
            }
        }
        queue.Add(root);
        return queue;
    }

    private List<ComputationalNode> TopologicalSort() {
        var list = new List<ComputationalNode>();
        var visited = new HashSet<ComputationalNode>();
        foreach (var node in _nodeMap.Keys) {
            if (!visited.Contains(node)) {
                var queue = Sort(node, visited);
                while (queue.Count != 0) {
                    list.Add(queue[0]);
                    queue.RemoveAt(0);
                }
            }
        }
        return list;
    }

    private void Clear() {
        foreach (var node in _nodeMap.Keys) {
            if (!node.IsLearnable()) {
                node.SetValue(null);
            }
            node.SetBackward(null);
            for (var i = 0; i < _nodeMap[node].Count; i++) {
                if (!node.IsLearnable()) {
                    _nodeMap[node][i].SetValue(null);
                }
                _nodeMap[node][i].SetBackward(null);
            }
        }
    }

    private void Update(HashSet<ComputationalNode> visited, ComputationalNode node) {
        visited.Add(node);
        if (node.IsLearnable()) {
            node.UpdateValue();
        }
        if (_nodeMap.ContainsKey(node)) {
            for (var i = 0; i < _nodeMap[node].Count; i++) {
                if (!visited.Contains(_nodeMap[node][i])) {
                    Update(visited, _nodeMap[node][i]);
                }
            }
        }
    }

    private void UpdateValues() {
        var visited = new HashSet<ComputationalNode>();
        foreach (var node in _nodeMap.Keys) {
            if (!visited.Contains(node)) {
                Update(visited, node);
            }
        }
    }

    private Matrix? CalculateDerivative(ComputationalNode node, ComputationalNode child) {
        var left = _reverseNodeMap[child][0];
        if (_reverseNodeMap[child].Count == 1) {
            IFunction function;
            switch (child.GetFunctionType()) {
                case FunctionType.SIGMOID:
                    function = new Sigmoid();
                    return child.GetBackward()?.ElementProduct(function.Derivative(child.GetValue()));
                case FunctionType.TANH:
                    function = new Tanh();
                    return child.GetBackward()?.ElementProduct(function.Derivative(child.GetValue()));
                case FunctionType.RELU:
                    function = new ReLU();
                    return child.GetBackward()?.ElementProduct(function.Derivative(child.GetValue()));
                case FunctionType.SOFTMAX:
                    function = new Softmax();
                    return child.GetBackward()?.ElementProduct(function.Derivative(child.GetValue()));
                default:
                    return null;
            }
        } else {
            var right = _reverseNodeMap[child][1];
            switch (child.GetOperator()) {
                case '*':
                    if (left.Equals(node)) {
                        return child.GetBackward()?.Multiply(right.GetValue()?.Transpose());
                    }
                    return left.GetValue()?.Transpose().Multiply(child.GetBackward());
                case '+':
                    return (Matrix?) child.GetBackward()?.Clone();
                case '-':
                    if (left.Equals(node)) {
                        return (Matrix?) child.GetBackward()?.Clone();
                    }
                    var result = (Matrix?) child.GetBackward()?.Clone();
                    for (var i = 0; i < result?.GetRow(); i++) {
                        for (var j = 0; j < result.GetColumn(); j++) {
                            result.SetValue(i, j, -result.GetValue(i, j));
                        }
                    }
                    return result;
            }
        }
        return null;
    }

    private void CalculateRMinusY(ComputationalNode output, double learningRate, List<int> classLabelIndex) {
        var backward = new Matrix(output.GetValue().GetRow(), output.GetValue().GetColumn());
        for (var i = 0; i < output.GetValue()?.GetRow(); i++) {
            for (var j = 0; j < output.GetValue()?.GetColumn(); j++) {
                if (classLabelIndex[i].Equals(j)) {
                    backward.SetValue(i, j, (1 - output.GetValue().GetValue(i, j)) * learningRate);
                } else {
                    backward.SetValue(i, j, (-output.GetValue().GetValue(i, j)) * learningRate);
                }
            }
        }
        output.SetBackward(backward);
    }

    public void Backpropagation(double learningRate, List<int> classLabelIndex) {
        var sortedNodes = TopologicalSort();
        var output = sortedNodes[0];
        sortedNodes.RemoveAt(0);
        CalculateRMinusY(output, learningRate, classLabelIndex);
        sortedNodes[0].SetBackward((Matrix?) output.GetBackward()?.Clone());
        sortedNodes.RemoveAt(0);
        while (sortedNodes.Count != 0) {
            var node = sortedNodes[0];
            sortedNodes.RemoveAt(0);
            for (var i = 0; i < _nodeMap[node].Count; i++) {
                var child = _nodeMap[node][i];
                if (node.GetBackward() == null) {
                    node.SetBackward(CalculateDerivative(node, child));
                } else {
                    node.GetBackward()?.Add(CalculateDerivative(node, child));
                }
            }
        }
        UpdateValues();
        Clear();
    }

    public List<int> ForwardCalculation() {
        var sortedNodes = TopologicalSort();
        var output = sortedNodes[0];
        while (sortedNodes.Count != 1) {
            var currentNode = sortedNodes[sortedNodes.Count - 1];
            sortedNodes.RemoveAt(sortedNodes.Count - 1);
            for (var i = 0; i < _nodeMap[currentNode].Count; i++) {
                var child = _nodeMap[currentNode][i];
                if (child.GetValue() == null) {
                    if (!child.GetFunctionType().Equals(null)) {
                        IFunction function;
                        switch (child.GetFunctionType()) {
                            case FunctionType.TANH:
                                function = new Tanh();
                                child.SetValue(function.Calculate(currentNode.GetValue()));
                                break;
                            case FunctionType.SIGMOID:
                                function = new Sigmoid();
                                child.SetValue(function.Calculate(currentNode.GetValue()));
                                break;
                            case FunctionType.RELU:
                                function = new ReLU();
                                child.SetValue(function.Calculate(currentNode.GetValue()));
                                break;
                            case FunctionType.SOFTMAX:
                                function = new Softmax();
                                child.SetValue(function.Calculate(currentNode.GetValue()));
                                break;
                        }
                    } else {
                        child.SetValue((Matrix?) currentNode.GetValue()?.Clone());
                    }
                } else {
                    if (child.GetFunctionType().Equals(null)) {
                        Matrix? result;
                        switch (child.GetOperator()) {
                            case '*':
                                if (child.GetValue()?.GetColumn() == currentNode.GetValue()?.GetRow()) {
                                    child.SetValue(child.GetValue()?.Multiply(currentNode.GetValue()));
                                } else {
                                    child.SetValue(currentNode.GetValue()?.Multiply(child.GetValue()));
                                }
                                break;
                            case '+':
                                result = (Matrix?) child.GetValue()?.Clone();
                                result?.Add(currentNode.GetValue());
                                child.SetValue(result);
                                break;
                            case '-':
                                result = (Matrix?) child.GetValue()?.Clone();
                                result?.Subtract(currentNode.GetValue());
                                child.SetValue(result);
                                break;
                        }
                    }
                }
            }
        }
        var classLabelIndex = new List<int>();
        for (var i = 0; i < output.GetValue()?.GetRow(); i++) {
            var max = Double.MinValue;
            var labelIndex = -1;
            for (var j = 0; j < output.GetValue()?.GetColumn(); j++) {
                if (max < output.GetValue()?.GetValue(i, j)) {
                    max = output.GetValue().GetValue(i, j);
                    labelIndex = j;
                }
            }
            classLabelIndex.Add(labelIndex);
        }
        return classLabelIndex;
    }
}