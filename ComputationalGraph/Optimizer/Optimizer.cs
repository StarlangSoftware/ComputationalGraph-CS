using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public abstract class Optimizer
    {
        protected double learningRate;
        private readonly double etaDecrease;

        public Optimizer(double learningRate, double etaDecrease)
        {
            this.learningRate = learningRate;
            this.etaDecrease = etaDecrease;
        }

        /// <summary>
        /// Updates the learning rate of the optimizer.
        /// </summary>
        public virtual void setLearningRate()
        {
            this.learningRate *= this.etaDecrease;
        }

        /// <summary>
        /// Checks if broadcasting be applied to the corresponding node.
        /// </summary>
        /// <param name="node">The node to check.</param>
        /// <returns>
        /// The index of the dimension where broadcasting is to be applied.
        /// -1 if broadcasting is not to be applied.
        /// </returns>
        private int broadcast(ComputationalNode node)
        {
            int[] v = node.getValue().GetShape();
            int[] b = node.getBackward().GetShape();
            int index = -1;

            for (int i = 0; i < v.Length; i++)
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
                    else
                    {
                        throw new ArgumentException("Value and Backward shapes are not compatible");
                    }
                }
            }

            return index;
        }

        /// <summary>
        /// Recursive helper function to update the values of learnable nodes.
        /// </summary>
        /// <param name="visited">A set of visited nodes.</param>
        /// <param name="node">The current node being processed.</param>
        private void updateRecursive(HashSet<ComputationalNode> visited, ComputationalNode node)
        {
            visited.Add(node);

            if (node.isLearnable())
            {
                int index = broadcast(node);

                if (index != -1)
                {
                    int v = 1;
                    int b = 1;

                    for (int i = node.getValue().GetShape().Length - 1; i >= index; i--)
                    {
                        v *= node.getValue().GetShape()[i];
                        b *= node.getBackward().GetShape()[i];
                    }

                    List<double> backwardValues = (List<double>)node.getBackward().GetData();
                    double[] values = new double[((List<double>)node.getValue().GetData()).Count];

                    for (int i = 0; i < backwardValues.Count; i++)
                    {
                        for (int j = i; j < i + b; j++)
                        {
                            values[((j - i) % v) + v * (j / b)] += backwardValues[j];
                        }

                        i += b - 1;
                    }

                    List<double> list = new List<double>();
                    foreach (double d in values)
                    {
                        list.Add(d);
                    }

                    node.setBackward(new Tensor(list, node.getValue().GetShape()));
                }

                this.setGradients(node);
                node.updateValue();
            }

            for (int t = 0; t < node.childrenSize(); t++)
            {
                ComputationalNode child = node.getChild(t);
                if (!visited.Contains(child))
                {
                    updateRecursive(visited, child);
                }
            }
        }

        /// <summary>
        /// Sets the gradients (backward values) of the node.
        /// </summary>
        /// <param name="node">The node whose gradients are to be set.</param>
        protected abstract void setGradients(ComputationalNode node);

        /// <summary>
        /// Updates the values of all learnable nodes in the graph.
        /// </summary>
        /// <param name="leafNodes">input nodes of the graph.</param>
        public virtual void updateValues(List<ComputationalNode> leafNodes)
        {
            HashSet<ComputationalNode> visited = new HashSet<ComputationalNode>();

            foreach (ComputationalNode node in leafNodes)
            {
                if (!visited.Contains(node))
                {
                    updateRecursive(visited, node);
                }
            }
        }
    }
}