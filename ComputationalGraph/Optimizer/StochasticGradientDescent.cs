using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public class StochasticGradientDescent : Optimizer
    {
        /**
         * <summary>Creates a stochastic gradient descent optimizer.</summary>
         *
         * <param name="learningRate">Initial learning rate.</param>
         * <param name="etaDecrease">Learning rate decay factor.</param>
         */
        public StochasticGradientDescent(double learningRate, double etaDecrease)
            : base(learningRate, etaDecrease)
        {
        }

        /**
         * <summary>Sets the gradients of the node as the learning-rate-scaled backward values.</summary>
         *
         * <param name="node">The node whose gradients are to be set.</param>
         */
        protected override void SetGradients(ComputationalNode node)
        {
            var values = new List<double>();
            var backward = (List<double>)node.GetBackward().GetData();

            foreach (var item in backward)
            {
                values.Add(item * LearningRate);
            }

            node.SetBackward(new Tensor(values, node.GetBackward().GetShape()));
        }
    }
}