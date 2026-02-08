using System;
using System.Collections.Generic;
using Classification.Performance;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using Math;

namespace ComputationalGraph;

public abstract class ComputationalGraph {
    
    private readonly Dictionary<ComputationalNode, List<ComputationalNode>> _nodeMap;
    private readonly Dictionary<ComputationalNode, List<ComputationalNode>> _reverseNodeMap;
    protected List<ComputationalNode> InputNodes;

    /**
     * <summary>Trains the computational graph using the given training set and parameters.</summary>
     * <param name="trainSet"> The training set.</param>
     * <param name="parameters"> The parameters of the computational graph.</param>
     */
    public abstract void Train(List<Tensor> trainSet, NeuralNetworkParameter parameters);
    
    /**
     * <summary>Tests the computational graph on the given test set.</summary>
     * <param name="testSet"> The test set.</param>
     * <returns> The classification performance of the computational graph on the test set.</returns>
     */
    public abstract ClassificationPerformance Test(List<Tensor> testSet);

    /**
     * <summary>Retrieves the class label indexes associated with the given output node in the computational graph.</summary>
     * <param name="outputNode"> The output node for which the class label indexes are to be retrieved.</param>
     * <returns> A list of integers representing the class label indexes.</returns>
     */
    protected abstract List<int> GetClassLabels(ComputationalNode outputNode);
    
    public ComputationalGraph() {
        _nodeMap = new Dictionary<ComputationalNode, List<ComputationalNode>>();
        _reverseNodeMap = new Dictionary<ComputationalNode, List<ComputationalNode>>();
        InputNodes = new List<ComputationalNode>();
    }

    private void ComputeIfAbsent(Dictionary<ComputationalNode, List<ComputationalNode>> dictionary,
        ComputationalNode first, ComputationalNode second)
    {
        if (!dictionary.ContainsKey(first))
        {
            dictionary[first] = [];
        }
        dictionary[first].Add(second);
    }

    public ComputationalNode AddEdge(ComputationalNode first, Object second, bool isBiased) {
        ComputationalNode newNode = null;
        if (second is Function.Function function)
        {
            newNode = new ComputationalNode(false, function, isBiased);
        }
        else
        {
            if (second is MultiplicationNode multiplicationNode)
            {
                newNode = new MultiplicationNode(false, isBiased, multiplicationNode.IsHadamard(), first);
            }
        }
        ComputeIfAbsent(_nodeMap, first, newNode);
        ComputeIfAbsent(_reverseNodeMap, newNode, first);
        if (second is MultiplicationNode node)
        {
            ComputeIfAbsent(_nodeMap, node, newNode);
            ComputeIfAbsent(_reverseNodeMap, newNode, node);
        }
        return newNode;
    }

    public ComputationalNode AddEdge(ComputationalNode first, ComputationalNode second, bool isBiased, bool isHadamard) {
        var newNode = new MultiplicationNode(false, isBiased, isHadamard, first);
        ComputeIfAbsent(_nodeMap, first, newNode);
        ComputeIfAbsent(_reverseNodeMap, newNode, first);
        ComputeIfAbsent(_nodeMap, second, newNode);
        ComputeIfAbsent(_reverseNodeMap, newNode, second);
        return newNode;
    }

    public ComputationalNode AddAdditionEdge(ComputationalNode first, ComputationalNode second, bool isBiased)
    {
        var newNode = new ComputationalNode(false, null, isBiased);
        ComputeIfAbsent(_nodeMap, first, newNode);
        ComputeIfAbsent(_reverseNodeMap, newNode, first);
        ComputeIfAbsent(_nodeMap, second, newNode);
        ComputeIfAbsent(_reverseNodeMap, newNode, second);
        return newNode;
    }

    /**
     * <summary>Concatenates the given nodes along the given dimension.</summary>
     * <param name="nodes"> List of nodes to be concatenated.</param>
     * <param name="dimension"> Dimension along which the nodes need to be concatenated.</param>
     * <returns> A new node that connects to the given nodes.</returns>
     */
    protected ComputationalNode ConcatEdges(List<ComputationalNode> nodes, int dimension)
    {
        var newNode = new ConcatenatedNode(dimension);
        foreach (var node in nodes)
        {
            ComputeIfAbsent(_nodeMap, node, newNode);
            ComputeIfAbsent(_reverseNodeMap, newNode, node);
            newNode.AddNode(node);
        }
        return newNode;
    }

    /**
     * <summary>Recursive helper function to perform depth-first search for topological sorting.</summary>
     * <param name="node"> The current node being processed.</param>
     * <param name="visited"> A set of visited nodes.</param>
     * <returns> A list representing the partial topological order.</returns>
     */
    private List<ComputationalNode> SortRecursive(ComputationalNode node, HashSet<ComputationalNode> visited) {
        var queue = new List<ComputationalNode>();
        visited.Add(node);
        if (_nodeMap.ContainsKey(node)) {
            foreach (var child in _nodeMap[node])
            {
                if (!visited.Contains(child)) {
                    queue.AddRange(SortRecursive(child, visited));
                }
            }
        }
        queue.Add(node);
        return queue;
    }

    /**
     * <summary>Performs topological sorting on the computational graph.</summary>
     * <returns> A list representing the topological order of the nodes.</returns>
     */
    private List<ComputationalNode> TopologicalSort() {
        var sortedList = new List<ComputationalNode>();
        var visited = new HashSet<ComputationalNode>();
        foreach (var node in _nodeMap.Keys) {
            if (!visited.Contains(node)) {
                var queue = SortRecursive(node, visited);
                while (queue.Count != 0) {
                    sortedList.Add(queue[0]);
                    queue.RemoveAt(0);
                }
            }
        }
        return sortedList;
    }

    /**
     * <summary>Recursive helper function to clear the values and gradients of nodes.</summary>
     */
    private void ClearRecursive(HashSet<ComputationalNode> visited, ComputationalNode node)
    {
        visited.Add(node);
        if (!node.IsLearnable()) {
            node.SetValue(null);
        }
        node.SetBackward(null);
        if (_nodeMap.ContainsKey(node))
        {
            foreach (var child in _nodeMap[node])
            {
                if (!visited.Contains(child))
                {
                    ClearRecursive(visited, child);
                }
            }
        }
    }
    
    /**
     *<summary>Clears the values and gradients of all nodes in the graph.</summary>
     */
    private void Clear() {
        var visited = new HashSet<ComputationalNode>();
        foreach (var node in _nodeMap.Keys) {
            if (!visited.Contains(node))
            {
                ClearRecursive(visited, node);
            }
        }
    }

    /**
     * <summary>Swaps the last two dimensions of the Tensor.</summary>
     * <param name="length"> dimension size.</param>
     */
    private int[] TransposeAxes(int length)
    {
        var axes = new int[length];
        for (var i = 0; i < length - 2; i++)
        {
            axes[i] = i;
        }
        axes[length - 1] = length - 2;
        axes[length - 2] = length - 1;
        return axes;
    }

    /**
     * <summary>Removes the bias term from the tensor.</summary>
     * <param name="tensor"> for which the bias term needs to be removed.</param>
     * <returns> Tensor without bias term.</returns>
     */
    private Tensor GetBiasedPartial(Tensor tensor)
    {
        var endIndexes = new int[tensor.GetShape().Length];
        for (int i = 0; i < endIndexes.Length; i++)
        {
            if (i == endIndexes.Length - 1)
            {
                endIndexes[i] = tensor.GetShape()[i] - 1;
            }
            else
            {
                endIndexes[i] = tensor.GetShape()[i];
            }
        }
        return tensor.Partial(new int[tensor.GetShape().Length], endIndexes);
    }

    /**
     * <summary>Calculates the derivative of the child node with respect to the parent node.</summary>
     * <param name="node"> Parent node.</param>
     * <param name="child"> Child node.</param>
     * <returns> The gradient tensor.</returns>
     */
    private Tensor CalculateDerivative(ComputationalNode node, ComputationalNode child) {
        var reverseChildren = _reverseNodeMap[child];
        if (reverseChildren == null || reverseChildren.Count == 0)
        {
            return null;
        }
        Tensor backward;
        if (child.IsBiased())
        {
            backward = GetBiasedPartial(child.GetBackward());
        }
        else
        {
            backward = child.GetBackward();
        }

        if (child.GetFunction() != null)
        {
            var function = child.GetFunction();
            Tensor childValue;
            if (child.IsBiased())
            {
                childValue = GetBiasedPartial(child.GetValue());
            }
            else
            {
                childValue = child.GetValue();
            }
            return function.Derivative(childValue, backward);
        }
        else
        {
            if (child is ConcatenatedNode concatenatedNode)
            {
                var index = concatenatedNode.GetIndex(node);
                var blockSize = backward.GetShape()[concatenatedNode.GetDimension()] / reverseChildren.Count;
                var dimensions = blockSize;
                var shape = new int[backward.GetShape().Length];
                for (var i = 0; i < backward.GetShape().Length; i++)
                {
                    if (concatenatedNode.GetDimension() > i)
                    {
                        shape[i] = backward.GetShape()[i];
                    }
                    else
                    {
                        if (concatenatedNode.GetDimension() < i)
                        {
                            dimensions *= backward.GetShape()[i];
                            shape[i] = backward.GetShape()[i];
                        }
                        else
                        {
                            shape[i] = blockSize;
                        }
                    }
                }

                var childValues = backward.GetData();
                var newValues = new List<double>();
                var j = index * dimensions;
                while (j < childValues.Count)
                {
                    for (var k = 0; k < dimensions; k++)
                    {
                        newValues.Add(childValues[j + k]);
                    }
                    j += reverseChildren.Count * dimensions;
                }
                return new Tensor(newValues, shape);
            }
            else
            {
                if (child is MultiplicationNode multiplicationNode)
                {
                    var left = reverseChildren[0];
                    var right = reverseChildren[1];
                    if (left == node)
                    {
                        var rightValue = right.GetValue();
                        if (multiplicationNode.IsHadamard())
                        {
                            return rightValue.HadamardProduct(backward);
                        }
                        return backward.Multiply(rightValue.Transpose(TransposeAxes(rightValue.GetShape().Length)));
                    }
                    var leftValue = left.GetValue();
                    if (multiplicationNode.IsHadamard())
                    {
                        return leftValue.HadamardProduct(backward);
                    }

                    if (leftValue != null && backward != null)
                    {
                        return leftValue.Transpose(TransposeAxes(leftValue.GetShape().Length)).Multiply(backward);
                    }
                }
                return backward;
            }
        }
    }

    /**
     * <summary>Computes the difference between the predicted and actual values (R - Y).</summary>
     * <param name="output"> The output node of the computational graph.</param>
     * <param name="classLabelIndex"> A list of true class labels (index of the correct class for each sample).</param>
     */
    private void CalculateRMinusY(ComputationalNode output, List<int> classLabelIndex) {
        var values = new List<double>();
        var outputValues = output.GetValue().GetData();
        var lastDimension = output.GetValue().GetShape()[output.GetValue().GetShape().Length - 1];
        for (var i = 0; i < outputValues.Count; i++) {
            if (i % lastDimension == classLabelIndex[i / lastDimension])
            {
                values.Add(1 - outputValues[i]);
            }
            else
            {
                values.Add(-outputValues[i]);
            }
        }
        var backward = new Tensor(values, output.GetValue().GetShape());
        output.SetBackward(backward);
    }

    /**
     * <summary>Performs backpropagation on the computational graph.</summary>
     * <param name="optimizer"> Optimizer to be used for updating the values.</param>
     * <param name="classLabelIndex"> The true class labels (as a list of integers).</param>
     */
    public void Backpropagation(Optimizer.Optimizer optimizer, List<int> classLabelIndex) {
        var sortedNodes = TopologicalSort();
        if (sortedNodes.Count == 0)
        {
            return;
        }
        var outputNode = sortedNodes[0];
        sortedNodes.RemoveAt(0);
        CalculateRMinusY(outputNode, classLabelIndex);
        if (sortedNodes.Count != 0)
        {
            sortedNodes[0].SetBackward(outputNode.GetBackward());
            sortedNodes.RemoveAt(0);
        }
        while (sortedNodes.Count != 0) {
            var node = sortedNodes[0];
            sortedNodes.RemoveAt(0);
            var children = _nodeMap[node];
            if (children != null)
            {
                foreach (var child in children) {
                    var derivative = CalculateDerivative(node, child);
                    if (derivative != null)
                    {
                        if (node.GetBackward() == null)
                        {
                            node.SetBackward(derivative);
                        }
                        else
                        {
                            node.SetBackward(node.GetBackward().Add(derivative));
                        }
                    }
                }
            }
        }
        optimizer.UpdateValues(_nodeMap);
        Clear();
    }

    /**
     * <summary>Add a bias term to the node's value by appending a column of ones.</summary>
     * <param name="tensor"> The node whose value needs to be biased.</param>
     */
    private void GetBiased(ComputationalNode tensor)
    {
        var lastDimensionSize = tensor.GetValue().GetShape()[tensor.GetValue().GetShape().Length - 1];
        var values = new List<double>();
        var oldValues = tensor.GetValue().GetData();
        for (var i = 0; i < oldValues.Count; i++)
        {
            values.Add(oldValues[i]);
            if ((i + 1) % lastDimensionSize == 0)
            {
                values.Add(1.0);
            }
        }
        var shape = new int[tensor.GetValue().GetShape().Length];
        for (var i = 0; i < shape.Length; i++)
        {
            if (i == shape.Length - 1)
            {
                shape[i] = tensor.GetValue().GetShape()[i] + 1;
            }
            else
            {
                shape[i] = tensor.GetValue().GetShape()[i];
            }
        }
        var biasedValue = new Tensor(values, shape);
        tensor.SetValue(biasedValue);
    }

    /**
     * <summary>Perform a forward pass and return predicted class indices.</summary>
     * <returns> A list of predicted class indices.</returns>
     */
    protected List<int> Predict()
    {
        var classLabels = ForwardCalculation(false);
        Clear();
        return classLabels;
    }

    /**
     * <summary>Perform a forward pass for the training phase.</summary>
     * <returns> A list of predicted class indices.</returns>
     */
    protected List<int> ForwardCalculation()
    {
        return ForwardCalculation(true);
    }

    /**
     * <summary>Perform a forward pass through the computational graph.</summary>
     * <param name="enableDropout"> Whether to enable dropout or not.</param>
     * <returns> A list of predicted class indices.</returns>
     */
    protected List<int> ForwardCalculation(bool enableDropout)
    {
        var sortedNodes = TopologicalSort();
        if (sortedNodes.Count == 0)
        {
            return [];
        }

        var outputNode = sortedNodes[0];
        var concatenatedNodeMap = new Dictionary<ComputationalNode, ComputationalNode[]>();
        var counterMap = new Dictionary<ComputationalNode, int>();
        while (sortedNodes.Count > 1)
        {
            var currentNode = sortedNodes[^1];
            sortedNodes.RemoveAt(sortedNodes.Count - 1);
            if (currentNode.IsBiased())
            {
                GetBiased(currentNode);
            }
            var children = _nodeMap[currentNode];
            if (children != null)
            {
                foreach (var child in children)
                {
                    if (child.GetValue() == null)
                    {
                        if (child.GetFunction() != null)
                        {
                            var function = child.GetFunction();
                            var currentValue = currentNode.GetValue();
                            if (function is Dropout dropout)
                            {
                                if (enableDropout)
                                {
                                    child.SetValue(dropout.Calculate(currentValue));
                                }
                                else
                                {
                                    child.SetValue(new Tensor(currentValue.GetData(),  currentValue.GetShape()));
                                }
                            }
                            else
                            {
                                child.SetValue(function.Calculate(currentValue));
                            }
                        }
                        else
                        {
                            if (child is ConcatenatedNode concatenatedNode)
                            {
                                if (!concatenatedNodeMap.ContainsKey(concatenatedNode))
                                {
                                    concatenatedNodeMap[concatenatedNode] = new ComputationalNode[_reverseNodeMap[child].Count];
                                }

                                concatenatedNodeMap[concatenatedNode][concatenatedNode.GetIndex(currentNode)] =
                                    currentNode;
                                if (!counterMap.ContainsKey(concatenatedNode))
                                {
                                    counterMap[concatenatedNode] = 0;
                                }
                                counterMap[concatenatedNode]++;
                                if (_reverseNodeMap[concatenatedNode].Count == counterMap[concatenatedNode])
                                {
                                    concatenatedNode.SetValue(concatenatedNodeMap[concatenatedNode][0].GetValue());
                                    for (var i = 1; i < concatenatedNodeMap[concatenatedNode].Length; i++)
                                    {
                                        concatenatedNode.SetValue(concatenatedNode.GetValue().Concat(concatenatedNodeMap[concatenatedNode][i].GetValue(), concatenatedNode.GetDimension()));
                                    }
                                }
                            }
                            else
                            {
                                child.SetValue(currentNode.GetValue());
                            }
                        }
                    }
                    else
                    {
                        if (child is MultiplicationNode multiplicationNode)
                        {
                            var childValue = child.GetValue();
                            var currentValue = currentNode.GetValue();
                            if (multiplicationNode.IsHadamard())
                            {
                                multiplicationNode.SetValue(childValue.HadamardProduct(currentValue));
                            }
                            else
                            {
                                if (!multiplicationNode.GetPriorityNode().Equals(currentNode))
                                {
                                    multiplicationNode.SetValue(childValue.Multiply(currentValue));
                                }
                                else
                                {
                                    multiplicationNode.SetValue(currentValue.Multiply(childValue));
                                }
                            }
                        }
                        else
                        {
                            var result =  child.GetValue();
                            var currentValue = currentNode.GetValue();
                            child.SetValue(result.Add(currentValue));
                        }
                    }
                }
            }
        }
        return GetClassLabels(outputNode);
    }
}