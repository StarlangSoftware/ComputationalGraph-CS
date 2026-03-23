using System;
using System.Collections.Generic;
using Classification.Performance;
using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using Math;

namespace Test
{
    [Serializable]
    public class LinearPerceptronSingleInput : ComputationalGraph.ComputationalGraph
    {
        /**
         * <summary>Creates a linear perceptron with a single input configuration.</summary>
         *
         * <param name="parameters">Neural network parameters.</param>
         */
        public LinearPerceptronSingleInput(NeuralNetworkParameter parameters)
            : base(parameters)
        {
        }

        /**
         * <summary>Creates the input tensor from the given instance by excluding the class label.</summary>
         *
         * <param name="instance">Input instance tensor.</param>
         * <returns>Input tensor without the class label.</returns>
         */
        private Tensor CreateInputTensor(Tensor instance)
        {
            var data = new List<double>();

            for (var i = 0; i < instance.GetShape()[0] - 1; i++)
            {
                data.Add(instance.GetValue(new[] { i }));
            }

            return new Tensor(data, new[] { 1, instance.GetShape()[0] - 1 });
        }

        /**
         * <summary>Trains the linear perceptron model.</summary>
         *
         * <param name="trainSet">Training set.</param>
         */
        public override void Train(List<Tensor> trainSet)
        {
            var input = new MultiplicationNode(false, true, false);
            InputNodes.Add(input);

            var weightsTensor = new Tensor(new List<double> { 1.0, 1.0, 1.0, 1.0 }, new[] { 2, 2 });
            var weightsNode = new MultiplicationNode(weightsTensor);

            var activationNode = AddEdge(input, weightsNode, false);
            OutputNode = AddEdge(activationNode, new Softmax(), false);

            var dataTensor = new Tensor(new List<double> { 1.0, 1.0 }, new[] { 2 });
            input.SetValue(CreateInputTensor(dataTensor));

            ForwardCalculation();
            Backpropagation();
        }

        /**
         * <summary>Tests the linear perceptron model.</summary>
         *
         * <param name="testSet">Test set.</param>
         * <returns>Classification performance of the model.</returns>
         */
        public override ClassificationPerformance Test(List<Tensor> testSet)
        {
            return null;
        }

        /**
         * <summary>Returns the output values of the given output node.</summary>
         *
         * <param name="outputNode">Output node of the graph.</param>
         * <returns>Output values of the node.</returns>
         */
        protected override List<double> GetOutputValue(ComputationalNode outputNode)
        {
            return null;
        }
    }
}