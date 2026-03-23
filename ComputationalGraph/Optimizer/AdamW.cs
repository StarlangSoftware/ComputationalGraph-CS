using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public class AdamW : Adam
    {
        private readonly double _weightDecay;

        /**
         * <summary>Creates an AdamW optimizer with the given hyperparameters.</summary>
         *
         * <param name="learningRate">Initial learning rate.</param>
         * <param name="etaDecrease">Learning rate decay factor.</param>
         * <param name="beta1">Exponential decay rate for the first moment estimates.</param>
         * <param name="beta2">Exponential decay rate for the second moment estimates.</param>
         * <param name="epsilon">Small constant for numerical stability.</param>
         * <param name="weightDecay">Weight decay coefficient.</param>
         */
        public AdamW(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon, double weightDecay)
            : base(learningRate, etaDecrease, beta1, beta2, epsilon)
        {
            _weightDecay = weightDecay;
        }

        /**
         * <summary>Sets the gradients for the given node using the AdamW optimization algorithm.</summary>
         *
         * <param name="node">The node whose gradients are to be set.</param>
         */
        protected override void SetGradients(ComputationalNode node)
        {
            var gradients = Calculate(node);
            var values = (List<double>)node.GetValue().GetData();

            for (var i = 0; i < gradients.Count; i++)
            {
                gradients[i] = gradients[i] + (LearningRate * _weightDecay * values[i]);
            }

            node.SetBackward(new Tensor(gradients, node.GetBackward().GetShape()));
        }
    }
}