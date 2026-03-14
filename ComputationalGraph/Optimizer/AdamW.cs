using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public class AdamW : Adam
    {
        private readonly double weightDecay;

        public AdamW(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon, double weightDecay)
            : base(learningRate, etaDecrease, beta1, beta2, epsilon)
        {
            this.weightDecay = weightDecay;
        }

        /// <summary>
        /// Sets the gradients for the given node using the AdamW optimization algorithm.
        /// </summary>
        /// <param name="node">The node whose gradients are to be set.</param>
        protected override void setGradients(ComputationalNode node)
        {
            List<double> gradients = calculate(node);
            List<double> values = (List<double>)node.getValue().GetData();

            for (int i = 0; i < gradients.Count; i++)
            {
                gradients[i] = gradients[i] + (learningRate * weightDecay * values[i]);
            }

            node.setBackward(new Tensor(gradients, node.getBackward().GetShape()));
        }
    }
}