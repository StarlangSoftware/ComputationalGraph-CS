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
    public class NeuralNetwork : ComputationalGraph.ComputationalGraph
    {
        /**
         * <summary>Creates a neural network with the given parameters.</summary>
         *
         * <param name="parameters">Neural network parameters.</param>
         */
        public NeuralNetwork(NeuralNetworkParameter parameters)
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
         * <summary>Creates a one-hot class label tensor.</summary>
         *
         * <param name="n">Number of classes.</param>
         * <param name="classLabel">Class label index.</param>
         * <returns>One-hot encoded class label tensor.</returns>
         */
        private Tensor SetClassLabelNode(int n, int classLabel)
        {
            var data = new List<double>();

            for (var i = 0; i < n; i++)
            {
                if (i == classLabel)
                {
                    data.Add(1.0);
                }
                else
                {
                    data.Add(0.0);
                }
            }

            return new Tensor(data, new[] { 1, n });
        }

        /**
         * <summary>Trains the neural network using the given training set.</summary>
         *
         * <param name="trainSet">Training set.</param>
         */
        public override void Train(List<Tensor> trainSet)
        {
            var input = new MultiplicationNode(false, true);
            var classLabelNode = new ComputationalNode();

            InputNodes.Add(input);
            InputNodes.Add(classLabelNode);

            var numberOfInputUnitsWithBiased = 5;
            var numberOfHiddenUnitsInLayer1 = 4;
            var t1 = new Tensor(
                Parameters.InitializeWeights(
                    numberOfInputUnitsWithBiased,
                    numberOfHiddenUnitsInLayer1,
                    new Random(Parameters.GetSeed())),
                new[] { numberOfInputUnitsWithBiased, numberOfHiddenUnitsInLayer1 });
            var w1 = new MultiplicationNode(t1);
            var a1 = AddEdge(input, w1);
            var a1Sigmoid = AddEdge(a1, new Sigmoid());
            var a1SigmoidDropout = AddEdge(
                a1Sigmoid,
                new Dropout(Parameters.GetDropout(), new Random(Parameters.GetSeed())),
                true);

            var numberOfHiddenUnitsInLayer2 = 20;
            var t2 = new Tensor(
                Parameters.InitializeWeights(
                    numberOfHiddenUnitsInLayer1 + 1,
                    numberOfHiddenUnitsInLayer2,
                    new Random(Parameters.GetSeed())),
                new[] { numberOfHiddenUnitsInLayer1 + 1, numberOfHiddenUnitsInLayer2 });
            var w2 = new MultiplicationNode(t2);
            var a2 = AddEdge(a1SigmoidDropout, w2);
            var a2Elu = AddEdge(a2, new ELU(3.0));
            var a2EluDropout = AddEdge(
                a2Elu,
                new Dropout(Parameters.GetDropout(), new Random(Parameters.GetSeed())),
                true);

            var numberOfClasses = 3;
            var t3 = new Tensor(
                Parameters.InitializeWeights(
                    numberOfHiddenUnitsInLayer2 + 1,
                    numberOfClasses,
                    new Random(Parameters.GetSeed())),
                new[] { numberOfHiddenUnitsInLayer2 + 1, numberOfClasses });
            var w3 = new MultiplicationNode(t3);
            var a3 = AddEdge(a2EluDropout, w3);

            OutputNode = AddEdge(a3, new Softmax());

            var nodes = new List<ComputationalNode>
            {
                OutputNode,
                classLabelNode
            };

            AddFunctionEdge(nodes, Parameters.GetLossFunction(), false);

            for (var i = 0; i < Parameters.GetEpoch(); i++)
            {
                var random = new Random(Parameters.GetSeed());

                for (var j = 0; j < trainSet.Count; j++)
                {
                    var i1 = random.Next(trainSet.Count);
                    var i2 = random.Next(trainSet.Count);

                    var temporary = trainSet[i1];
                    trainSet[i1] = trainSet[i2];
                    trainSet[i2] = temporary;
                }

                foreach (var instance in trainSet)
                {
                    input.SetValue(CreateInputTensor(instance));
                    classLabelNode.SetValue(
                        SetClassLabelNode(
                            numberOfClasses,
                            (int)instance.GetValue(new[] { instance.GetShape()[0] - 1 })));

                    ForwardCalculation();
                    Backpropagation();
                }

                Parameters.GetOptimizer().SetLearningRate();
            }
        }

        /**
         * <summary>Tests the neural network using the given test set.</summary>
         *
         * <param name="testSet">Test set.</param>
         * <returns>Classification performance of the model.</returns>
         */
        public override ClassificationPerformance Test(List<Tensor> testSet)
        {
            var count = 0;
            var total = 0;

            foreach (var instance in testSet)
            {
                InputNodes[0].SetValue(CreateInputTensor(instance));

                var classLabel = (int)Predict()[0];
                if (classLabel == (int)instance.GetValue(new[] { instance.GetShape()[0] - 1 }))
                {
                    count++;
                }

                total++;
            }

            return new ClassificationPerformance((count + 0.0) / total);
        }

        /**
         * <summary>Returns the predicted output value as a class label index.</summary>
         *
         * <param name="outputNode">Output node of the network.</param>
         * <returns>Predicted class label index as a list.</returns>
         */
        protected override List<double> GetOutputValue(ComputationalNode outputNode)
        {
            var classLabelIndices = new List<double>();
            var outputValue = outputNode.GetValue();

            if (outputValue != null)
            {
                var columnCount = outputValue.GetShape()[1];
                var maxValue = double.NegativeInfinity;
                var labelIndex = -1;

                for (var j = 0; j < columnCount; j++)
                {
                    var value = outputValue.GetValue(new[] { 0, j });
                    if (maxValue < value)
                    {
                        maxValue = value;
                        labelIndex = j;
                    }
                }

                classLabelIndices.Add(labelIndex + 0.0);
            }

            return classLabelIndices;
        }
    }
}