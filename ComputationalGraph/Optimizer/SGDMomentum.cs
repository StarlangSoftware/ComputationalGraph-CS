using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public class SGDMomentum : Optimizer
    {
        protected readonly Dictionary<ComputationalNode, double[]> VelocityMap;
        protected readonly double Momentum;

        /**
         * <summary>Creates an SGD optimizer with momentum.</summary>
         *
         * <param name="learningRate">Initial learning rate.</param>
         * <param name="etaDecrease">Learning rate decay factor.</param>
         * <param name="momentum">Momentum coefficient.</param>
         */
        public SGDMomentum(double learningRate, double etaDecrease, double momentum)
            : base(learningRate, etaDecrease)
        {
            VelocityMap = new Dictionary<ComputationalNode, double[]>();
            Momentum = momentum;
        }

        /**
         * <summary>Calculates momentum-adjusted gradients for the given node.</summary>
         *
         * <param name="node">The node whose gradients are to be set.</param>
         */
        protected override void SetGradients(ComputationalNode node)
        {
            var backwardSize = ((List<double>)node.GetBackward().GetData()).Count;
            var newValues = new List<double>(backwardSize);

            for (var i = 0; i < backwardSize; i++)
            {
                newValues.Add((1 - Momentum) * ((List<double>)node.GetBackward().GetData())[i]);
            }

            if (VelocityMap.ContainsKey(node))
            {
                for (var i = 0; i < newValues.Count; i++)
                {
                    newValues[i] = newValues[i] + (VelocityMap[node][i] * Momentum);
                }
            }

            var velocity = new double[backwardSize];
            for (var i = 0; i < backwardSize; i++)
            {
                velocity[i] = newValues[i];
            }

            VelocityMap[node] = velocity;

            for (var i = 0; i < newValues.Count; i++)
            {
                newValues[i] = newValues[i] * LearningRate;
            }

            node.SetBackward(new Tensor(newValues, node.GetBackward().GetShape()));
        }
    }
}