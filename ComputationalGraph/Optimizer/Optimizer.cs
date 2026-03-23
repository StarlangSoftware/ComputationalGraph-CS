using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public abstract class Optimizer
    {
        protected double LearningRate;
        private readonly double _etaDecrease;

        /**
         * <summary>Creates an optimizer with the given learning rate and decay factor.</summary>
         *
         * <param name="learningRate">Initial learning rate.</param>
         * <param name="etaDecrease">Learning rate decay factor.</param>
         */
        public Optimizer(double learningRate, double etaDecrease)
        {
            LearningRate = learningRate;
            _etaDecrease = etaDecrease;
        }

        /**
         * <summary>Updates the learning rate of the optimizer.</summary>
         */
        public virtual void SetLearningRate()
        {
            LearningRate *= _etaDecrease;
        }

        /**
         * <summary>Checks if broadcasting should be applied to the corresponding node.</summary>
         *
         * <param name="node">The node to check.</param>
         * <returns>
         * The index of the dimension where broadcasting is to be applied,
         * or -1 if broadcasting is not to be applied.
         * </returns>
         */
        private int Broadcast(ComputationalNode node)
        {
            var valueShape = node.GetValue().GetShape();
            var backwardShape = node.GetBackward().GetShape();
            var index = -1;

            for (var i = 0; i < valueShape.Length; i++)
            {
                if (valueShape[i] != backwardShape[i])
                {
                    if (valueShape[i] == 1)
                    {
                        if (index != -1)
                        {
                            return -1;
                        }

                        index = i;
                    }
                    else
                    {
                        throw new ArgumentException("Value and backward shapes are not compatible");
                    }
                }
            }

            return index;
        }

        /**
         * <summary>Recursively updates the values of learnable nodes.</summary>
         *
         * <param name="visited">A set of visited nodes.</param>
         * <param name="node">The current node being processed.</param>
         */
        private void UpdateRecursive(HashSet<ComputationalNode> visited, ComputationalNode node)
        {
            visited.Add(node);

            if (node.IsLearnable())
            {
                var index = Broadcast(node);

                if (index != -1)
                {
                    var valueBlockSize = 1;
                    var backwardBlockSize = 1;

                    for (var i = node.GetValue().GetShape().Length - 1; i >= index; i--)
                    {
                        valueBlockSize *= node.GetValue().GetShape()[i];
                        backwardBlockSize *= node.GetBackward().GetShape()[i];
                    }

                    var backwardValues = (List<double>)node.GetBackward().GetData();
                    var values = new double[((List<double>)node.GetValue().GetData()).Count];

                    for (var i = 0; i < backwardValues.Count; i++)
                    {
                        for (var j = i; j < i + backwardBlockSize; j++)
                        {
                            values[((j - i) % valueBlockSize) + valueBlockSize * (j / backwardBlockSize)] += backwardValues[j];
                        }

                        i += backwardBlockSize - 1;
                    }

                    var list = new List<double>();
                    foreach (var value in values)
                    {
                        list.Add(value);
                    }

                    node.SetBackward(new Tensor(list, node.GetValue().GetShape()));
                }

                SetGradients(node);
                node.UpdateValue();
            }

            for (var t = 0; t < node.ChildrenSize(); t++)
            {
                var child = node.GetChild(t);
                if (!visited.Contains(child))
                {
                    UpdateRecursive(visited, child);
                }
            }
        }

        /**
         * <summary>Sets the gradients of the given node.</summary>
         *
         * <param name="node">The node whose gradients are to be set.</param>
         */
        protected abstract void SetGradients(ComputationalNode node);

        /**
         * <summary>Updates the values of all learnable nodes in the graph.</summary>
         *
         * <param name="leafNodes">Input nodes of the graph.</param>
         */
        public virtual void UpdateValues(List<ComputationalNode> leafNodes)
        {
            var visited = new HashSet<ComputationalNode>();

            foreach (var node in leafNodes)
            {
                if (!visited.Contains(node))
                {
                    UpdateRecursive(visited, node);
                }
            }
        }
    }
}