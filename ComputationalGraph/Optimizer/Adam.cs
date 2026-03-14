using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public class Adam : SGDMomentum
    {
        private readonly Dictionary<ComputationalNode, double[]> momentumMap;
        private readonly double beta2;
        private readonly double epsilon;
        private double currentBeta1;
        private double currentBeta2;

        public Adam(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon)
            : base(learningRate, etaDecrease, beta1)
        {
            this.momentumMap = new Dictionary<ComputationalNode, double[]>();
            this.beta2 = beta2;
            this.epsilon = epsilon;
            this.currentBeta1 = 1;
            this.currentBeta2 = 1;
        }

        /// <summary>
        /// Calculates the gradient updates using the Adam optimization algorithm.
        /// </summary>
        /// <param name="node">The node whose gradients are to be set.</param>
        protected List<double> calculate(ComputationalNode node)
        {
            int backwardSize = ((List<double>)node.getBackward().GetData()).Count;

            List<double> newValuesMomentum = new List<double>(backwardSize);
            List<double> newValuesVelocity = new List<double>(backwardSize);

            for (int i = 0; i < backwardSize; i++)
            {
                double backwardValue = ((List<double>)node.getBackward().GetData())[i];
                newValuesMomentum.Add((1 - momentum) * backwardValue);
                newValuesVelocity.Add((1 - beta2) * (backwardValue * backwardValue));
            }

            if (momentumMap.ContainsKey(node))
            {
                for (int i = 0; i < newValuesVelocity.Count; i++)
                {
                    newValuesVelocity[i] = newValuesVelocity[i] + beta2 * velocityMap[node][i];
                    newValuesMomentum[i] = newValuesMomentum[i] + momentum * momentumMap[node][i];
                }
            }

            double[] momentumValues = new double[backwardSize];
            double[] velocityValues = new double[backwardSize];

            for (int i = 0; i < backwardSize; i++)
            {
                momentumValues[i] = newValuesMomentum[i];
                velocityValues[i] = newValuesVelocity[i];
            }

            momentumMap[node] = momentumValues;
            velocityMap[node] = velocityValues;

            for (int i = 0; i < newValuesMomentum.Count; i++)
            {
                newValuesMomentum[i] = newValuesMomentum[i] / (1 - this.currentBeta1);
            }

            for (int i = 0; i < newValuesVelocity.Count; i++)
            {
                newValuesVelocity[i] = newValuesVelocity[i] / (1 - this.currentBeta2);
            }

            List<double> newValues = new List<double>(newValuesMomentum.Count);
            for (int i = 0; i < newValuesMomentum.Count; i++)
            {
                newValues.Add(
                    (newValuesMomentum[i] / (System.Math.Sqrt(newValuesVelocity[i]) + epsilon)) * learningRate
                );
            }

            return newValues;
        }

        /// <summary>
        /// Sets the gradients for the given node using the Adam optimization algorithm.
        /// </summary>
        /// <param name="node">The node whose gradients are to be set.</param>
        protected override void setGradients(ComputationalNode node)
        {
            node.setBackward(new Tensor(calculate(node), node.getBackward().GetShape()));
        }

        /// <summary>
        /// Updates the values of all learnable nodes and momentum values of the graph.
        /// </summary>
        /// <param name="leafNodes">input nodes of the graph.</param>
        public override void updateValues(List<ComputationalNode> leafNodes)
        {
            this.currentBeta1 *= momentum;
            this.currentBeta2 *= beta2;
            base.updateValues(leafNodes);
        }
    }
}