using System;
using System.Collections.Generic;
using System.IO;
using System.Runtime.Serialization.Formatters.Binary;
using Classification.Performance;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;
using CGFunction = ComputationalGraph.Function.Function;
using Dropout = ComputationalGraph.Function.Dropout;

namespace ComputationalGraph
{
    [Serializable]
    public abstract class ComputationalGraph
    {
        protected ComputationalNode OutputNode;
        protected List<ComputationalNode> InputNodes;
        private List<ComputationalNode> _leafNodes;
        protected readonly NeuralNetworkParameter Parameters;

        /**
         * <summary>Creates a computational graph with the given neural network parameters.</summary>
         *
         * <param name="parameters">Neural network parameters of the graph.</param>
         */
        public ComputationalGraph(NeuralNetworkParameter parameters)
        {
            InputNodes = new List<ComputationalNode>();
            Parameters = parameters;
        }

        /**
         * <summary>Trains the computational graph using the given training set.</summary>
         *
         * <param name="trainSet">Training set used for training.</param>
         */
        public abstract void Train(List<Tensor> trainSet);

        /**
         * <summary>Tests the computational graph using the given test set.</summary>
         *
         * <param name="testSet">Test set used for evaluation.</param>
         * <returns>Classification performance of the model.</returns>
         */
        public abstract ClassificationPerformance Test(List<Tensor> testSet);

        /**
         * <summary>Returns the output values of the given output node.</summary>
         *
         * <param name="outputNode">Output node of the graph.</param>
         * <returns>Output values of the node.</returns>
         */
        protected abstract List<double> GetOutputValue(ComputationalNode outputNode);

        /**
         * <summary>Adds an edge between the first node and the second object.</summary>
         *
         * <param name="first">First node.</param>
         * <param name="second">Second object which may be a function or computational node.</param>
         * <param name="isBiased">Indicates whether the new node is biased.</param>
         * <returns>The newly created computational node.</returns>
         */
        protected ComputationalNode AddEdge(ComputationalNode first, object second, bool isBiased)
        {
            if (second is CGFunction function)
            {
                var nodes = new List<ComputationalNode>();
                nodes.Add(first);
                return function.AddEdge(nodes, isBiased);
            }

            var newNode = second is MultiplicationNode multiplicationNode
                ? new MultiplicationNode(false, isBiased, multiplicationNode.IsHadamard(), first)
                : throw new ArgumentException("Illegal Type of Object: second");

            first.AddChild(newNode);
            newNode.AddParent(first);
            ((ComputationalNode)second).AddChild(newNode);
            newNode.AddParent((ComputationalNode)second);
            return newNode;
        }

        /**
         * <summary>Adds an edge between the first node and the second object without bias.</summary>
         *
         * <param name="first">First node.</param>
         * <param name="second">Second object which may be a function or computational node.</param>
         * <returns>The newly created computational node.</returns>
         */
        protected ComputationalNode AddEdge(ComputationalNode first, object second)
        {
            return AddEdge(first, second, false);
        }

        /**
         * <summary>Adds a function edge for the given input nodes.</summary>
         *
         * <param name="inputNodes">Input nodes of the function.</param>
         * <param name="second">Function to be applied.</param>
         * <param name="isBiased">Indicates whether the new node is biased.</param>
         * <returns>The newly created computational node.</returns>
         */
        protected ComputationalNode AddFunctionEdge(List<ComputationalNode> inputNodes, CGFunction second, bool isBiased)
        {
            return second.AddEdge(inputNodes, isBiased);
        }

        /**
         * <summary>Adds a multiplication edge between two computational nodes.</summary>
         *
         * <param name="first">First node.</param>
 <param name="first">First node.</param>
         * <param name="second">Second node.</param>
         * <param name="isBiased">Indicates whether the new node is biased.</param>
         * <param name="isHadamard">Indicates whether the multiplication is Hadamard multiplication.</param>
         * <returns>The newly created multiplication node.</returns>
         */
        protected ComputationalNode AddEdge(ComputationalNode first, ComputationalNode second, bool isBiased, bool isHadamard)
        {
            var newNode = new MultiplicationNode(false, isBiased, isHadamard, first);
            first.AddChild(newNode);
            newNode.AddParent(first);
            second.AddChild(newNode);
            newNode.AddParent(second);
            return newNode;
        }

        /**
         * <summary>Adds an addition edge between two computational nodes.</summary>
         *
         * <param name="first">First node.</param>
         * <param name="second">Second node.</param>
         * <param name="isBiased">Indicates whether the new node is biased.</param>
         * <returns>The newly created addition node.</returns>
         */
        protected ComputationalNode AddAdditionEdge(ComputationalNode first, ComputationalNode second, bool isBiased)
        {
            var newNode = new ComputationalNode(false, isBiased);
            first.AddChild(newNode);
            newNode.AddParent(first);
            second.AddChild(newNode);
            newNode.AddParent(second);
            return newNode;
        }

        /**
         * <summary>Concatenates the given nodes along the specified dimension.</summary>
         *
         * <param name="nodes">Nodes to be concatenated.</param>
         * <param name="dimension">Concatenation dimension.</param>
         * <returns>The concatenated node.</returns>
         */
        protected ComputationalNode ConcatEdges(List<ComputationalNode> nodes, int dimension)
        {
            var newNode = new ConcatenatedNode(dimension);
            foreach (var node in nodes)
            {
                node.AddChild(newNode);
                newNode.AddParent(node);
                newNode.AddNode(node);
            }

            return newNode;
        }

        /**
         * <summary>Recursively sorts nodes in topological order.</summary>
         *
         * <param name="node">Current node.</param>
         * <param name="visited">Visited node set.</param>
         * <returns>Sorted linked list of nodes.</returns>
         */
        private LinkedList<ComputationalNode> SortRecursive(ComputationalNode node, HashSet<ComputationalNode> visited)
        {
            var queue = new LinkedList<ComputationalNode>();
            visited.Add(node);

            for (var i = 0; i < node.ChildrenSize(); i++)
            {
                var child = node.GetChild(i);
                if (!visited.Contains(child))
                {
                    var childQueue = SortRecursive(child, visited);
                    foreach (var item in childQueue)
                    {
                        queue.AddLast(item);
                    }
                }
            }

            queue.AddLast(node);
            return queue;
        }

        /**
         * <summary>Returns the topologically sorted list of nodes.</summary>
         *
         * <returns>Topologically sorted linked list of nodes.</returns>
         */
        private LinkedList<ComputationalNode> TopologicalSort()
        {
            var sortedList = new LinkedList<ComputationalNode>();
            var visited = new HashSet<ComputationalNode>();

            foreach (var node in _leafNodes)
            {
                if (!visited.Contains(node))
                {
                    var queue = SortRecursive(node, visited);
                    while (queue.Count > 0)
                    {
                        sortedList.AddLast(queue.First.Value);
                        queue.RemoveFirst();
                    }
                }
            }

            return sortedList;
        }

        /**
         * <summary>Clears values and backward tensors recursively starting from the given node.</summary>
         *
         * <param name="visited">Visited node set.</param>
         * <param name="node">Current node.</param>
         */
        private void ClearRecursive(HashSet<ComputationalNode> visited, ComputationalNode node)
        {
            visited.Add(node);

            if (!node.IsLearnable())
            {
                node.SetValue(null);
            }

            node.SetBackward(null);

            for (var i = 0; i < node.ChildrenSize(); i++)
            {
                var child = node.GetChild(i);
                if (!visited.Contains(child))
                {
                    ClearRecursive(visited, child);
                }
            }
        }

        /**
         * <summary>Clears temporary values and backward tensors in the graph.</summary>
         */
        private void Clear()
        {
            var visited = new HashSet<ComputationalNode>();
            foreach (var node in _leafNodes)
            {
                if (!visited.Contains(node))
                {
                    ClearRecursive(visited, node);
                }
            }
        }

        /**
         * <summary>Returns the transposed axis order for the given tensor rank.</summary>
         *
         * <param name="length">Rank of the tensor.</param>
         * <returns>Axis order for transpose operation.</returns>
         */
        private int[] TransposeAxes(int length)
        {
            var axes = new int[length];
            for (var i = 0; i < axes.Length - 2; i++)
            {
                axes[i] = i;
            }

            axes[axes.Length - 1] = axes.Length - 2;
            axes[axes.Length - 2] = axes.Length - 1;
            return axes;
        }

        /**
         * <summary>Returns the unbiased partial tensor by removing the last biased component.</summary>
         *
         * <param name="tensor">Input tensor.</param>
         * <returns>Unbiased partial tensor.</returns>
         */
        private Tensor GetBiasedPartial(Tensor tensor)
        {
            var endIndexes = new int[tensor.GetShape().Length];
            for (var i = 0; i < endIndexes.Length; i++)
            {
                endIndexes[i] = i == endIndexes.Length - 1
                    ? tensor.GetShape()[i] - 1
                    : tensor.GetShape()[i];
            }

            return tensor.Partial(new int[tensor.GetShape().Length], endIndexes);
        }

        /**
         * <summary>Calculates the derivative of the given child node with respect to the given node.</summary>
         *
         * <param name="node">Parent node.</param>
         * <param name="child">Child node.</param>
         * <returns>Calculated derivative tensor.</returns>
         */
        private Tensor CalculateDerivative(ComputationalNode node, ComputationalNode child)
        {
            if (child.ParentsSize() == 0)
            {
                return null;
            }

            var backward = child.IsBiasedNode()
                ? GetBiasedPartial(child.GetBackward())
                : child.GetBackward();

            if (child is FunctionNode functionNode)
            {
                var function = functionNode.GetFunction();
                var childValue = child.IsBiasedNode()
                    ? GetBiasedPartial(child.GetValue())
                    : child.GetValue();

                return function.Derivative(childValue, backward);
            }

            if (child is ConcatenatedNode concatenatedNode)
            {
                var index = concatenatedNode.GetIndex(node);
                var blockSize = backward.GetShape()[concatenatedNode.GetDimension()] / child.ParentsSize();
                var dimensions = blockSize;
                var shape = new int[backward.GetShape().Length];

                for (var i = 0; i < backward.GetShape().Length; i++)
                {
                    if (concatenatedNode.GetDimension() > i)
                    {
                        shape[i] = backward.GetShape()[i];
                    }
                    else if (concatenatedNode.GetDimension() < i)
                    {
                        dimensions *= backward.GetShape()[i];
                        shape[i] = backward.GetShape()[i];
                    }
                    else
                    {
                        shape[i] = blockSize;
                    }
                }

                var childValues = (List<double>)backward.GetData();
                var newValues = new List<double>();

                var start = index * dimensions;
                while (start < childValues.Count)
                {
                    for (var k = 0; k < dimensions; k++)
                    {
                        newValues.Add(childValues[start + k]);
                    }

                    start += child.ParentsSize() * dimensions;
                }

                return new Tensor(newValues, shape);
            }

            if (child is MultiplicationNode multiplicationNode)
            {
                var left = child.GetParent(0);
                var right = child.GetParent(1);

                if (ReferenceEquals(left, node))
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

                throw new NullReferenceException("Backward and/or left child values are null");
            }

            return backward;
        }

        /**
         * <summary>Performs backpropagation on the graph.</summary>
         */
        protected void Backpropagation()
        {
            var sortedNodes = TopologicalSort();
            if (sortedNodes.Count == 0)
            {
                return;
            }

            var currentOutputNode = sortedNodes.First.Value;
            sortedNodes.RemoveFirst();

            var backward = new List<double>();
            for (var i = 0; i < ((List<double>)currentOutputNode.GetValue().GetData()).Count; i++)
            {
                backward.Add(1.0 / Parameters.GetBatchSize());
            }

            currentOutputNode.SetBackward(new Tensor(backward, currentOutputNode.GetValue().GetShape()));

            while (sortedNodes.Count > 0)
            {
                var node = sortedNodes.First.Value;
                sortedNodes.RemoveFirst();

                if (node.ChildrenSize() > 0)
                {
                    for (var i = 0; i < node.ChildrenSize(); i++)
                    {
                        var child = node.GetChild(i);
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

            Parameters.GetOptimizer().UpdateValues(_leafNodes);
            Clear();
        }

        /**
         * <summary>Adds bias values to the given node tensor.</summary>
         *
         * <param name="tensor">Node whose tensor will be biased.</param>
         */
        private void GetBiased(ComputationalNode tensor)
        {
            var lastDimensionSize = tensor.GetValue().GetShape()[tensor.GetValue().GetShape().Length - 1];
            var values = new List<double>();
            var oldValues = (List<double>)tensor.GetValue().GetData();

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
                shape[i] = i == shape.Length - 1
                    ? tensor.GetValue().GetShape()[i] + 1
                    : tensor.GetValue().GetShape()[i];
            }

            var biasedValue = new Tensor(values, shape);
            tensor.SetValue(biasedValue);
        }

        /**
         * <summary>Predicts class labels using forward calculation without dropout.</summary>
         *
         * <returns>Predicted class labels.</returns>
         */
        protected List<double> Predict()
        {
            var classLabels = ForwardCalculation(false);
            Clear();
            return classLabels;
        }

        /**
         * <summary>Performs forward calculation with dropout enabled.</summary>
         *
         * <returns>Output class labels.</returns>
         */
        protected List<double> ForwardCalculation()
        {
            if (_leafNodes == null)
            {
                _leafNodes = FindLeafNodes();
            }

            return ForwardCalculation(true);
        }

        /**
         * <summary>Finds the output node starting from the given node.</summary>
         *
         * <param name="node">Starting node.</param>
         * <returns>Output node of the graph.</returns>
         */
        private ComputationalNode FindOutputNode(ComputationalNode node)
        {
            if (node.ChildrenSize() == 0)
            {
                return node;
            }

            return FindOutputNode(node.GetChild(0));
        }

        /**
         * <summary>Finds all leaf nodes in the graph.</summary>
         *
         * <returns>List of leaf nodes.</returns>
         */
        private List<ComputationalNode> FindLeafNodes()
        {
            var leafNodes = new List<ComputationalNode>();
            var foundOutputNode = FindOutputNode(InputNodes[0]);

            var queue = new List<ComputationalNode>();
            var visited = new HashSet<ComputationalNode>();

            queue.Add(foundOutputNode);

            while (queue.Count > 0)
            {
                var currentNode = queue[0];
                queue.RemoveAt(0);

                if (currentNode.ParentsSize() == 0)
                {
                    leafNodes.Add(currentNode);
                }

                for (var i = 0; i < currentNode.ParentsSize(); i++)
                {
                    var parent = currentNode.GetParent(i);
                    if (!visited.Contains(parent))
                    {
                        visited.Add(parent);
                        queue.Add(parent);
                    }
                }
            }

            return leafNodes;
        }

        /**
         * <summary>Performs forward calculation with optional dropout.</summary>
         *
         * <param name="enableDropout">Indicates whether dropout is enabled.</param>
         * <returns>Output class labels.</returns>
         */
        private List<double> ForwardCalculation(bool enableDropout)
        {
            var sortedNodes = TopologicalSort();
            if (sortedNodes.Count == 0)
            {
                return new List<double>();
            }

            var concatenatedNodeMap = new Dictionary<ComputationalNode, ComputationalNode[]>();
            var counterMap = new Dictionary<ComputationalNode, int>();

            while (sortedNodes.Count > 1)
            {
                var currentNode = sortedNodes.Last.Value;
                sortedNodes.RemoveLast();

                if (currentNode.IsBiasedNode())
                {
                    GetBiased(currentNode);
                }

                if (currentNode.GetValue() == null)
                {
                    throw new ArgumentException("Current node's value is null");
                }

                if (currentNode.ChildrenSize() > 0)
                {
                    if (ReferenceEquals(currentNode, OutputNode) && !enableDropout)
                    {
                        break;
                    }

                    for (var t = 0; t < currentNode.ChildrenSize(); t++)
                    {
                        var child = currentNode.GetChild(t);

                        if (child.GetValue() == null)
                        {
                            if (child is FunctionNode functionNode)
                            {
                                var function = functionNode.GetFunction();
                                var currentValue = currentNode.GetValue();

                                if (function is Dropout)
                                {
                                    if (enableDropout)
                                    {
                                        child.SetValue(function.Calculate(currentValue));
                                    }
                                    else
                                    {
                                        child.SetValue(new Tensor(currentValue.GetData(), currentValue.GetShape()));
                                    }
                                }
                                else
                                {
                                    child.SetValue(function.Calculate(currentValue));
                                }
                            }
                            else if (child is ConcatenatedNode concatenatedNode)
                            {
                                if (!concatenatedNodeMap.ContainsKey(child))
                                {
                                    concatenatedNodeMap[child] = new ComputationalNode[child.ParentsSize()];
                                }

                                concatenatedNodeMap[child][concatenatedNode.GetIndex(currentNode)] = currentNode;

                                if (!counterMap.ContainsKey(child))
                                {
                                    counterMap[child] = 0;
                                }

                                counterMap[child] = counterMap[child] + 1;

                                if (child.ParentsSize() == counterMap[child])
                                {
                                    child.SetValue(concatenatedNodeMap[child][0].GetValue());

                                    for (var i = 1; i < concatenatedNodeMap[child].Length; i++)
                                    {
                                        child.SetValue(
                                            child.GetValue().Concat(
                                                concatenatedNodeMap[child][i].GetValue(),
                                                concatenatedNode.GetDimension()));
                                    }
                                }
                            }
                            else
                            {
                                child.SetValue(currentNode.GetValue());
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
                                    child.SetValue(childValue.HadamardProduct(currentValue));
                                }
                                else if (!ReferenceEquals(multiplicationNode.GetPriorityNode(), currentNode))
                                {
                                    child.SetValue(childValue.Multiply(currentValue));
                                }
                                else
                                {
                                    child.SetValue(currentValue.Multiply(childValue));
                                }
                            }
                            else
                            {
                                var result = child.GetValue();
                                var currentValue = currentNode.GetValue();
                                child.SetValue(result.Add(currentValue));
                            }
                        }
                    }
                }
            }

            return GetOutputValue(OutputNode);
        }

#pragma warning disable SYSLIB0011
        /**
         * <summary>Saves the computational graph to the given file.</summary>
         *
         * <param name="fileName">Output file name.</param>
         */
        public void Save(string fileName)
        {
            try
            {
                using var outFile = new FileStream(fileName, FileMode.Create);
                var formatter = new BinaryFormatter();
                formatter.Serialize(outFile, this);
            }
            catch (IOException)
            {
                Console.WriteLine("Object could not be saved.");
            }
        }

        /**
         * <summary>Loads a computational graph model from the given file.</summary>
         *
         * <param name="fileName">Input file name.</param>
         * <returns>Loaded computational graph if successful; otherwise, null.</returns>
         */
        public static ComputationalGraph LoadModel(string fileName)
        {
            try
            {
                using var inFile = new FileStream(fileName, FileMode.Open);
                var formatter = new BinaryFormatter();
                return (ComputationalGraph)formatter.Deserialize(inFile);
            }
            catch (IOException)
            {
                return null;
            }
            catch (System.Runtime.Serialization.SerializationException)
            {
                return null;
            }
        }
#pragma warning restore SYSLIB0011
    }
}