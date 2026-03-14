using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public class StochasticGradientDescent : Optimizer
    {
        public StochasticGradientDescent(double learningRate, double etaDecrease)
            : base(learningRate, etaDecrease)
        {
        }

        /// <summary>
        /// Sets the gradients (backward values) of the node to the learning rate times the backward values.
        /// </summary>
        /// <param name="node">The node whose gradients are to be set.</param>
        protected override void setGradients(ComputationalNode node)
        {
            List<double> values = new List<double>();
            List<double> backward = (List<double>)node.getBackward().GetData();

            foreach (double item in backward)
            {
                values.Add(item * this.learningRate);
            }

            node.setBackward(new Tensor(values, node.getBackward().GetShape()));
        }
    }
}