using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using Math;

namespace Test;

public class LinearPerceptron : NeuralNetwork
{
    public override void Train(List<Tensor> trainSet, NeuralNetworkParameter parameters)
    {
        var optimizer = parameters.GetOptimizer();
        var input = new MultiplicationNode(false, true);
        InputNodes.Add(input);
        const int numberOfInputUnitsWithBiased = 5;
        const int numberOfClasses = 3;
        var initialization = parameters.GetInitialization();
        var initialWeights = initialization.Initialize(numberOfInputUnitsWithBiased, numberOfClasses, new Random(1));
        int[] weightsShape = [numberOfInputUnitsWithBiased, numberOfClasses];
        var weightsTensor = new Tensor(initialWeights, weightsShape);
        var w = new MultiplicationNode(weightsTensor);
        var a = AddEdge(input, w, false);
        var softmax = new Softmax();
        AddEdge(a, softmax, false);
        for (var i = 0; i < parameters.GetEpoch(); i++)
        {
            foreach (var instance in trainSet)
            {
                input.SetValue(CreateInputTensor(instance));
                var calculatedClassses = ForwardCalculation();
                int[] index = [instance.GetShape()[0] - 1];
                List<int> classes = [(int) instance.GetValue(index)];
                Backpropagation(optimizer, classes);
            }
        }
    }
}