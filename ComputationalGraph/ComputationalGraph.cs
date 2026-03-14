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
        protected ComputationalNode outputNode;
        protected List<ComputationalNode> inputNodes;
        private List<ComputationalNode> leafNodes;
        protected readonly NeuralNetworkParameter parameters;

        public ComputationalGraph(NeuralNetworkParameter parameters)
        {
            this.inputNodes = new List<ComputationalNode>();
            this.parameters = parameters;
        }

        public abstract void train(List<Tensor> trainSet);

        public abstract ClassificationPerformance test(List<Tensor> testSet);

        protected abstract List<double> getOutputValue(ComputationalNode outputNode);

        protected ComputationalNode addEdge(ComputationalNode first, object second, bool isBiased)
        {
            if (second is CGFunction function)
            {
                List<ComputationalNode> nodes = new List<ComputationalNode>();
                nodes.Add(first);
                return function.addEdge(nodes, isBiased);
            }
            else
            {
                ComputationalNode newNode;
                if (second is MultiplicationNode multiplicationNode)
                {
                    newNode = new MultiplicationNode(false, isBiased, multiplicationNode.isHadamard(), first);
                }
                else
                {
                    throw new ArgumentException("Illegal Type of Object: second");
                }

                first.addChild(newNode);
                newNode.addParent(first);
                ((ComputationalNode)second).addChild(newNode);
                newNode.addParent((ComputationalNode)second);
                return newNode;
            }
        }

        protected ComputationalNode addEdge(ComputationalNode first, object second)
        {
            return addEdge(first, second, false);
        }

        protected ComputationalNode addFunctionEdge(List<ComputationalNode> inputNodes, CGFunction second, bool isBiased)
        {
            return second.addEdge(inputNodes, isBiased);
        }

        protected ComputationalNode addEdge(ComputationalNode first, ComputationalNode second, bool isBiased, bool isHadamard)
        {
            ComputationalNode newNode = new MultiplicationNode(false, isBiased, isHadamard, first);
            first.addChild(newNode);
            newNode.addParent(first);
            second.addChild(newNode);
            newNode.addParent(second);
            return newNode;
        }

        protected ComputationalNode addAdditionEdge(ComputationalNode first, ComputationalNode second, bool isBiased)
        {
            ComputationalNode newNode = new ComputationalNode(false, isBiased);
            first.addChild(newNode);
            newNode.addParent(first);
            second.addChild(newNode);
            newNode.addParent(second);
            return newNode;
        }

        protected ComputationalNode concatEdges(List<ComputationalNode> nodes, int dimension)
        {
            ConcatenatedNode newNode = new ConcatenatedNode(dimension);
            foreach (ComputationalNode node in nodes)
            {
                node.addChild(newNode);
                newNode.addParent(node);
                newNode.addNode(node);
            }
            return newNode;
        }

        private LinkedList<ComputationalNode> sortRecursive(ComputationalNode node, HashSet<ComputationalNode> visited)
        {
            LinkedList<ComputationalNode> queue = new LinkedList<ComputationalNode>();
            visited.Add(node);

            for (int i = 0; i < node.childrenSize(); i++)
            {
                ComputationalNode child = node.getChild(i);
                if (!visited.Contains(child))
                {
                    LinkedList<ComputationalNode> childQueue = sortRecursive(child, visited);
                    foreach (ComputationalNode item in childQueue)
                    {
                        queue.AddLast(item);
                    }
                }
            }

            queue.AddLast(node);
            return queue;
        }

        private LinkedList<ComputationalNode> topologicalSort()
        {
            LinkedList<ComputationalNode> sortedList = new LinkedList<ComputationalNode>();
            HashSet<ComputationalNode> visited = new HashSet<ComputationalNode>();

            foreach (ComputationalNode node in leafNodes)
            {
                if (!visited.Contains(node))
                {
                    LinkedList<ComputationalNode> queue = sortRecursive(node, visited);
                    while (queue.Count > 0)
                    {
                        sortedList.AddLast(queue.First.Value);
                        queue.RemoveFirst();
                    }
                }
            }

            return sortedList;
        }

        private void clearRecursive(HashSet<ComputationalNode> visited, ComputationalNode node)
        {
            visited.Add(node);

            if (!node.isLearnable())
            {
                node.setValue(null);
            }

            node.setBackward(null);

            for (int i = 0; i < node.childrenSize(); i++)
            {
                ComputationalNode child = node.getChild(i);
                if (!visited.Contains(child))
                {
                    clearRecursive(visited, child);
                }
            }
        }

        private void clear()
        {
            HashSet<ComputationalNode> visited = new HashSet<ComputationalNode>();
            foreach (ComputationalNode node in leafNodes)
            {
                if (!visited.Contains(node))
                {
                    clearRecursive(visited, node);
                }
            }
        }

        private int[] transposeAxes(int length)
        {
            int[] axes = new int[length];
            for (int i = 0; i < axes.Length - 2; i++)
            {
                axes[i] = i;
            }

            axes[axes.Length - 1] = axes.Length - 2;
            axes[axes.Length - 2] = axes.Length - 1;
            return axes;
        }

        private Tensor getBiasedPartial(Tensor tensor)
        {
            int[] endIndexes = new int[tensor.GetShape().Length];
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

        private Tensor calculateDerivative(ComputationalNode node, ComputationalNode child)
        {
            if (child.parentsSize() == 0)
            {
                return null;
            }

            Tensor backward;
            if (child.isBiasedNode())
            {
                backward = getBiasedPartial(child.getBackward());
            }
            else
            {
                backward = child.getBackward();
            }

            if (child is FunctionNode functionNode)
            {
                CGFunction function = functionNode.getFunction();
                Tensor childValue;

                if (child.isBiasedNode())
                {
                    childValue = getBiasedPartial(child.getValue());
                }
                else
                {
                    childValue = child.getValue();
                }

                return function.derivative(childValue, backward);
            }
            else
            {
                if (child is ConcatenatedNode concatenatedNode)
                {
                    int index = concatenatedNode.getIndex(node);
                    int blockSize = backward.GetShape()[concatenatedNode.getDimension()] / child.parentsSize();
                    int dimensions = blockSize;
                    int[] shape = new int[backward.GetShape().Length];

                    for (int i = 0; i < backward.GetShape().Length; i++)
                    {
                        if (concatenatedNode.getDimension() > i)
                        {
                            shape[i] = backward.GetShape()[i];
                        }
                        else if (concatenatedNode.getDimension() < i)
                        {
                            dimensions *= backward.GetShape()[i];
                            shape[i] = backward.GetShape()[i];
                        }
                        else
                        {
                            shape[i] = blockSize;
                        }
                    }

                    List<double> childValues = (List<double>)backward.GetData();
                    List<double> newValues = new List<double>();

                    int start = index * dimensions;
                    while (start < childValues.Count)
                    {
                        for (int k = 0; k < dimensions; k++)
                        {
                            newValues.Add(childValues[start + k]);
                        }

                        start += child.parentsSize() * dimensions;
                    }

                    return new Tensor(newValues, shape);
                }
                else
                {
                    if (child is MultiplicationNode multiplicationNode)
                    {
                        ComputationalNode left = child.getParent(0);
                        ComputationalNode right = child.getParent(1);

                        if (ReferenceEquals(left, node))
                        {
                            Tensor rightValue = right.getValue();
                            if (multiplicationNode.isHadamard())
                            {
                                return rightValue.HadamardProduct(backward);
                            }

                            return backward.Multiply(rightValue.Transpose(transposeAxes(rightValue.GetShape().Length)));
                        }

                        Tensor leftValue = left.getValue();
                        if (multiplicationNode.isHadamard())
                        {
                            return leftValue.HadamardProduct(backward);
                        }

                        if (leftValue != null && backward != null)
                        {
                            return leftValue.Transpose(transposeAxes(leftValue.GetShape().Length)).Multiply(backward);
                        }

                        throw new NullReferenceException("Backward and/or left child values are null");
                    }

                    return backward;
                }
            }
        }

        protected void backpropagation()
        {
            LinkedList<ComputationalNode> sortedNodes = topologicalSort();
            if (sortedNodes.Count == 0)
            {
                return;
            }

            ComputationalNode currentOutputNode = sortedNodes.First.Value;
            sortedNodes.RemoveFirst();

            List<double> backward = new List<double>();
            for (int i = 0; i < ((List<double>)currentOutputNode.getValue().GetData()).Count; i++)
            {
                backward.Add(1.0 / this.parameters.getBatchSize());
            }

            currentOutputNode.setBackward(new Tensor(backward, currentOutputNode.getValue().GetShape()));

            while (sortedNodes.Count > 0)
            {
                ComputationalNode node = sortedNodes.First.Value;
                sortedNodes.RemoveFirst();

                if (node.childrenSize() > 0)
                {
                    for (int i = 0; i < node.childrenSize(); i++)
                    {
                        ComputationalNode child = node.getChild(i);
                        Tensor derivative = calculateDerivative(node, child);

                        if (derivative != null)
                        {
                            if (node.getBackward() == null)
                            {
                                node.setBackward(derivative);
                            }
                            else
                            {
                                node.setBackward(node.getBackward().Add(derivative));
                            }
                        }
                    }
                }
            }

            this.parameters.getOptimizer().updateValues(this.leafNodes);
            clear();
        }

        private void getBiased(ComputationalNode tensor)
        {
            int lastDimensionSize = tensor.getValue().GetShape()[tensor.getValue().GetShape().Length - 1];
            List<double> values = new List<double>();
            List<double> oldValues = (List<double>)tensor.getValue().GetData();

            for (int i = 0; i < oldValues.Count; i++)
            {
                values.Add(oldValues[i]);
                if ((i + 1) % lastDimensionSize == 0)
                {
                    values.Add(1.0);
                }
            }

            int[] shape = new int[tensor.getValue().GetShape().Length];
            for (int i = 0; i < shape.Length; i++)
            {
                if (i == shape.Length - 1)
                {
                    shape[i] = tensor.getValue().GetShape()[i] + 1;
                }
                else
                {
                    shape[i] = tensor.getValue().GetShape()[i];
                }
            }

            Tensor biasedValue = new Tensor(values, shape);
            tensor.setValue(biasedValue);
        }

        protected List<double> predict()
        {
            List<double> classLabels = forwardCalculation(false);
            clear();
            return classLabels;
        }

        protected List<double> forwardCalculation()
        {
            if (leafNodes == null)
            {
                leafNodes = findLeafNodes();
            }

            return forwardCalculation(true);
        }

        private ComputationalNode findOutputNode(ComputationalNode node)
        {
            if (node.childrenSize() == 0)
            {
                return node;
            }

            return findOutputNode(node.getChild(0));
        }

        private List<ComputationalNode> findLeafNodes()
        {
            List<ComputationalNode> leafNodes = new List<ComputationalNode>();
            ComputationalNode foundOutputNode = findOutputNode(inputNodes[0]);

            List<ComputationalNode> queue = new List<ComputationalNode>();
            HashSet<ComputationalNode> visited = new HashSet<ComputationalNode>();

            queue.Add(foundOutputNode);

            while (queue.Count > 0)
            {
                ComputationalNode currentNode = queue[0];
                queue.RemoveAt(0);

                if (currentNode.parentsSize() == 0)
                {
                    leafNodes.Add(currentNode);
                }

                for (int i = 0; i < currentNode.parentsSize(); i++)
                {
                    ComputationalNode parent = currentNode.getParent(i);
                    if (!visited.Contains(parent))
                    {
                        visited.Add(parent);
                        queue.Add(parent);
                    }
                }
            }

            return leafNodes;
        }

        private List<double> forwardCalculation(bool enableDropout)
        {
            LinkedList<ComputationalNode> sortedNodes = topologicalSort();
            if (sortedNodes.Count == 0)
            {
                return new List<double>();
            }

            Dictionary<ComputationalNode, ComputationalNode[]> concatenatedNodeMap =
                new Dictionary<ComputationalNode, ComputationalNode[]>();

            Dictionary<ComputationalNode, int> counterMap =
                new Dictionary<ComputationalNode, int>();

            while (sortedNodes.Count > 1)
            {
                ComputationalNode currentNode = sortedNodes.Last.Value;
                sortedNodes.RemoveLast();

                if (currentNode.isBiasedNode())
                {
                    getBiased(currentNode);
                }

                if (currentNode.getValue() == null)
                {
                    throw new ArgumentException("Current node's value is null");
                }

                if (currentNode.childrenSize() > 0)
                {
                    if (ReferenceEquals(currentNode, outputNode) && !enableDropout)
                    {
                        break;
                    }

                    for (int t = 0; t < currentNode.childrenSize(); t++)
                    {
                        ComputationalNode child = currentNode.getChild(t);

                        if (child.getValue() == null)
                        {
                            if (child is FunctionNode functionNode)
                            {
                                CGFunction function = functionNode.getFunction();
                                Tensor currentValue = currentNode.getValue();

                                if (function is Dropout)
                                {
                                    if (enableDropout)
                                    {
                                        child.setValue(function.calculate(currentValue));
                                    }
                                    else
                                    {
                                        child.setValue(new Tensor(currentValue.GetData(), currentValue.GetShape()));
                                    }
                                }
                                else
                                {
                                    child.setValue(function.calculate(currentValue));
                                }
                            }
                            else
                            {
                                if (child is ConcatenatedNode concatenatedNode)
                                {
                                    if (!concatenatedNodeMap.ContainsKey(child))
                                    {
                                        concatenatedNodeMap[child] = new ComputationalNode[child.parentsSize()];
                                    }

                                    concatenatedNodeMap[child][concatenatedNode.getIndex(currentNode)] = currentNode;

                                    if (!counterMap.ContainsKey(child))
                                    {
                                        counterMap[child] = 0;
                                    }

                                    counterMap[child] = counterMap[child] + 1;

                                    if (child.parentsSize() == counterMap[child])
                                    {
                                        child.setValue(concatenatedNodeMap[child][0].getValue());

                                        for (int i = 1; i < concatenatedNodeMap[child].Length; i++)
                                        {
                                            child.setValue(
                                                child.getValue().Concat(
                                                    concatenatedNodeMap[child][i].getValue(),
                                                    concatenatedNode.getDimension()));
                                        }
                                    }
                                }
                                else
                                {
                                    child.setValue(currentNode.getValue());
                                }
                            }
                        }
                        else
                        {
                            if (child is MultiplicationNode multiplicationNode)
                            {
                                Tensor childValue = child.getValue();
                                Tensor currentValue = currentNode.getValue();

                                if (multiplicationNode.isHadamard())
                                {
                                    child.setValue(childValue.HadamardProduct(currentValue));
                                }
                                else if (!ReferenceEquals(multiplicationNode.getPriorityNode(), currentNode))
                                {
                                    child.setValue(childValue.Multiply(currentValue));
                                }
                                else
                                {
                                    child.setValue(currentValue.Multiply(childValue));
                                }
                            }
                            else
                            {
                                Tensor result = child.getValue();
                                Tensor currentValue = currentNode.getValue();
                                child.setValue(result.Add(currentValue));
                            }
                        }
                    }
                }
            }

            return getOutputValue(outputNode);
        }

#pragma warning disable SYSLIB0011
        public void save(string fileName)
        {
            try
            {
                using FileStream outFile = new FileStream(fileName, FileMode.Create);
                BinaryFormatter formatter = new BinaryFormatter();
                formatter.Serialize(outFile, this);
            }
            catch (IOException)
            {
                Console.WriteLine("Object could not be saved.");
            }
        }

        public static ComputationalGraph loadModel(string fileName)
        {
            try
            {
                using FileStream inFile = new FileStream(fileName, FileMode.Open);
                BinaryFormatter formatter = new BinaryFormatter();
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