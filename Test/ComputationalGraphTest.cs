using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using Classification.Performance;
using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using ComputationalGraph.Optimizer;
using NUnit.Framework;
using Tensor = Math.Tensor;

namespace Test
{
    public class ComputationalGraphTest
    {
        /**
         * <summary>Tests the linear perceptron single input model.</summary>
         */
        
        [Test]
        public void LinearPerceptronSingleInputTest()
        {
            var graph = new LinearPerceptronSingleInput(
                new NeuralNetworkParameter(
                    1,
                    100,
                    new StochasticGradientDescent(0.1, 0.99)));

            Assert.That(() => graph.Train(new List<Tensor>()), Throws.Nothing);
        }

        /**
         * <summary>Tests the neural network on the Iris dataset.</summary>
         */
        [Test]
        public void NeuralNetworkTest()
        {
            var labelMap = new Dictionary<string, int>();
            var dataSet = new List<string[]>();
            var filePath = GetIrisFilePath();

            Assert.That(File.Exists(filePath), Is.True, "iris.txt could not be found.");

            foreach (var line in File.ReadAllLines(filePath))
            {
                var instance = line.Split(',');
                dataSet.Add(instance);

                var label = instance[instance.Length - 1];
                if (!labelMap.ContainsKey(label))
                {
                    labelMap[label] = labelMap.Count;
                }
            }

            Shuffle(dataSet, new Random(1));

            var trainList = new List<Tensor>();
            var testList = new List<Tensor>();

            for (var i = 0; i < dataSet.Count; i++)
            {
                var values = new List<double>();

                for (var j = 0; j < dataSet[i].Length - 1; j++)
                {
                    values.Add(double.Parse(dataSet[i][j], CultureInfo.InvariantCulture));
                }

                values.Add(labelMap[dataSet[i][dataSet[i].Length - 1]] + 0.0);

                if (i >= 120)
                {
                    testList.Add(new Tensor(values, new[] { values.Count }));
                }
                else
                {
                    trainList.Add(new Tensor(values, new[] { values.Count }));
                }
            }

            var graph = new NeuralNetwork(
                new NeuralNetworkParameter(
                    1,
                    100,
                    new StochasticGradientDescent(0.1, 0.99),
                    new CrossEntropyLoss(),
                    0.0));

            graph.Train(trainList);

            var performance = graph.Test(testList);

            Assert.That(performance, Is.Not.Null);
            Assert.That(performance.GetAccuracy(), Is.InRange(0.0, 1.0));
        }

        /**
         * <summary>Tests concatenation and feature propagation operations in the computational graph.</summary>
         */
        [Test]
        public void FeaturesTest()
        {
            var graph = new FeatureGraph(
                new NeuralNetworkParameter(
                    1,
                    1,
                    new StochasticGradientDescent(0.1, 0.99)));

            graph.Train(null);
        }

        /**
         * <summary>Returns the path of the Iris dataset file.</summary>
         *
         * <returns>The path of the Iris dataset file.</returns>
         */
        private static string GetIrisFilePath()
        {
            var candidatePaths = new[]
            {
                Path.Combine(TestContext.CurrentContext.TestDirectory, "iris.txt"),
                Path.Combine(TestContext.CurrentContext.WorkDirectory, "iris.txt"),
                "iris.txt"
            };

            foreach (var candidatePath in candidatePaths)
            {
                if (File.Exists(candidatePath))
                {
                    return candidatePath;
                }
            }

            return candidatePaths[0];
        }

        /**
         * <summary>Shuffles the given list using the provided random generator.</summary>
         *
         * <param name="list">List to be shuffled.</param>
         * <param name="random">Random generator.</param>
         */
        private static void Shuffle<T>(IList<T> list, Random random)
        {
            for (var i = list.Count - 1; i > 0; i--)
            {
                var j = random.Next(i + 1);
                var temporary = list[i];
                list[i] = list[j];
                list[j] = temporary;
            }
        }

        [Serializable]
        private sealed class FeatureGraph : ComputationalGraph.ComputationalGraph
        {
            /**
             * <summary>Creates a feature graph with the given parameters.</summary>
             *
             * <param name="parameters">Neural network parameters.</param>
             */
            
            public FeatureGraph(NeuralNetworkParameter parameters)
                : base(parameters)
            
            {

            }

            /**
             * <summary>Trains the feature graph and checks the expected output.</summary>
             *
             * <param name="trainSet">Training set.</param>
             */
            public override void Train(List<Tensor> trainSet)
            {
                var input = new MultiplicationNode(false, false);
                InputNodes.Add(input);

                input.SetValue(new Tensor(new List<double> { 1.0, 2.0, 3.0, 4.0 }, new[] { 2, 1, 2 }));

                
                
                var nodes = new List<ComputationalNode>();
                for (var i = 0; i < 4; i++)
                {
                    var weightNode = new MultiplicationNode(
                        new Tensor(
                            new List<double> { 6.0, 5.0, 4.0, 3.0, 2.0, 1.0 },
                            new[] { 1, 2, 3 }));

                    nodes.Add(AddEdge(input, weightNode));
                }

                var concatenatedNode = ConcatEdges(nodes, 1); // dim=1

                var outputWeightNode = new MultiplicationNode(
                    new Tensor(
                        new List<double> { 6.0, 5.0, 1.0 },
                        new[] { 1, 3, 1 }));

                OutputNode = AddEdge(concatenatedNode, outputWeightNode);

                ForwardCalculation();
                
                
                Backpropagation();
                
                
                input.SetValue(new Tensor(new List<double> { 4.0, 3.0, 2.0, 1.0 }, new[] { 2, 1, 2 }));
                ForwardCalculation();
              
                
                var output = (List<double>)OutputNode.GetValue().GetData();
                TestContext.WriteLine(string.Join(", ", output));
                var expected = new List<double>
                {
                    2202.44,
                    2202.44,
                    2202.44,
                    2202.44,
                    973.6400000000001,
                    973.6400000000001,
                    973.6400000000001,
                    973.6400000000001
                };

                Assert.That(output, Is.EqualTo(expected));
            }

            /**
             * <summary>Tests the feature graph.</summary>
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
                if (outputNode?.GetValue() == null) return null;
                return (List<double>)outputNode.GetValue().GetData();
            }
        }
    }
}