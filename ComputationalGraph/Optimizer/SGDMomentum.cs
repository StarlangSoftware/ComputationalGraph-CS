using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public class SGDMomentum : Optimizer
    {
        protected readonly Dictionary<ComputationalNode, double[]> velocityMap;
        protected readonly double momentum;

        public SGDMomentum(double learningRate, double etaDecrease, double momentum)
            : base(learningRate, etaDecrease)
        {
            this.velocityMap = new Dictionary<ComputationalNode, double[]>();
            this.momentum = momentum;
        }

        /// <summary>
        /// Calculates the new gradients by combining the current gradient with the previous velocity.
        /// It updates the internal velocity state and modifies the node's backward tensor
        /// to reflect the momentum-adjusted update step.
        /// </summary>
        /// <param name="node">The node whose gradients are to be set.</param>
        protected override void setGradients(ComputationalNode node)
        {
            int backwardSize = ((List<double>)node.getBackward().GetData()).Count;
            List<double> newValues = new List<double>(backwardSize);

            for (int i = 0; i < backwardSize; i++)
            {
                newValues.Add((1 - momentum) * ((List<double>)node.getBackward().GetData())[i]);
            }

            if (velocityMap.ContainsKey(node))
            {
                for (int i = 0; i < newValues.Count; i++)
                {
                    newValues[i] = newValues[i] + (velocityMap[node][i] * momentum);
                }
            }

            double[] velocity = new double[backwardSize];
            for (int i = 0; i < backwardSize; i++)
            {
                velocity[i] = newValues[i];
            }

            velocityMap[node] = velocity;

            for (int i = 0; i < newValues.Count; i++)
            {
                newValues[i] = newValues[i] * learningRate;
            }

            node.setBackward(new Tensor(newValues, node.getBackward().GetShape()));
        }
    }
}