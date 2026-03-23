using System;
using System.Collections.Generic;
using ComputationalGraph.Node;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Optimizer
{
    [Serializable]
    public class Adam : SGDMomentum
    {
        private readonly Dictionary<ComputationalNode, double[]> _momentumMap;
        private readonly double _beta2;
        private readonly double _epsilon;
        private double _currentBeta1;
        private double _currentBeta2;

        /**
         * <summary>Creates an Adam optimizer with the given hyperparameters.</summary>
         *
         * <param name="learningRate">Initial learning rate.</param>
         * <param name="etaDecrease">Learning rate decay factor.</param>
         * <param name="beta1">Exponential decay rate for the first moment estimates.</param>
         * <param name="beta2">Exponential decay rate for the second moment estimates.</param>
         * <param name="epsilon">Small constant for numerical stability.</param>
         */
        public Adam(double learningRate, double etaDecrease, double beta1, double beta2, double epsilon)
            : base(learningRate, etaDecrease, beta1)
        {
            _momentumMap = new Dictionary<ComputationalNode, double[]>();
            _beta2 = beta2;
            _epsilon = epsilon;
            _currentBeta1 = 1;
            _currentBeta2 = 1;
        }

        /**
         * <summary>Calculates gradient updates using the Adam optimization algorithm.</summary>
         *
         * <param name="node">The node whose gradients are to be calculated.</param>
         * <returns>A list of updated gradient values.</returns>
         */
        protected List<double> Calculate(ComputationalNode node)
        {
            var backwardSize = ((List<double>)node.GetBackward().GetData()).Count;

            var newValuesMomentum = new List<double>(backwardSize);
            var newValuesVelocity = new List<double>(backwardSize);

            for (var i = 0; i < backwardSize; i++)
            {
                var backwardValue = ((List<double>)node.GetBackward().GetData())[i];
                newValuesMomentum.Add((1 - Momentum) * backwardValue);
                newValuesVelocity.Add((1 - _beta2) * (backwardValue * backwardValue));
            }

            if (_momentumMap.ContainsKey(node))
            {
                for (var i = 0; i < newValuesVelocity.Count; i++)
                {
                    newValuesVelocity[i] = newValuesVelocity[i] + _beta2 * VelocityMap[node][i];
                    newValuesMomentum[i] = newValuesMomentum[i] + Momentum * _momentumMap[node][i];
                }
            }

            var momentumValues = new double[backwardSize];
            var velocityValues = new double[backwardSize];

            for (var i = 0; i < backwardSize; i++)
            {
                momentumValues[i] = newValuesMomentum[i];
                velocityValues[i] = newValuesVelocity[i];
            }

            _momentumMap[node] = momentumValues;
            VelocityMap[node] = velocityValues;

            for (var i = 0; i < newValuesMomentum.Count; i++)
            {
                newValuesMomentum[i] = newValuesMomentum[i] / (1 - _currentBeta1);
            }

            for (var i = 0; i < newValuesVelocity.Count; i++)
            {
                newValuesVelocity[i] = newValuesVelocity[i] / (1 - _currentBeta2);
            }

            var newValues = new List<double>(newValuesMomentum.Count);
            for (var i = 0; i < newValuesMomentum.Count; i++)
            {
                newValues.Add(
                    (newValuesMomentum[i] / (System.Math.Sqrt(newValuesVelocity[i]) + _epsilon)) * LearningRate
                );
            }

            return newValues;
        }

        /**
         * <summary>Sets the gradients for the given node using the Adam optimization algorithm.</summary>
         *
         * <param name="node">The node whose gradients are to be set.</param>
         */
        protected override void SetGradients(ComputationalNode node)
        {
            node.SetBackward(new Tensor(Calculate(node), node.GetBackward().GetShape()));
        }

        /**
         * <summary>Updates the values of all learnable nodes and Adam moment estimates.</summary>
         *
         * <param name="leafNodes">Input nodes of the graph.</param>
         */
        public override void UpdateValues(List<ComputationalNode> leafNodes)
        {
            _currentBeta1 *= Momentum;
            _currentBeta2 *= _beta2;
            base.UpdateValues(leafNodes);
        }
    }
}