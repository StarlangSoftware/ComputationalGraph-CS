using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using Math;

namespace Test;

public class DeepNetwork : NeuralNetwork
{
    public override void Train(List<Tensor> trainSet, NeuralNetworkParameter parameters)
    {
        var optimizer = parameters.GetOptimizer();
        var input = new MultiplicationNode(false, true);
        InputNodes.Add(input);
        const int numberOfInputUnitsWithBiased = 5;
        const int numberOfHiddenUnitsInLayer1 = 6;
        var initialization = parameters.GetInitialization();
        var initialWeights = initialization.Initialize(numberOfInputUnitsWithBiased, numberOfHiddenUnitsInLayer1, new Random(1));
        int[] weightsShape = [numberOfInputUnitsWithBiased, numberOfHiddenUnitsInLayer1];
        var weightsTensor = new Tensor(initialWeights, weightsShape);
        var w = new MultiplicationNode(weightsTensor);
        var a = AddEdge(input, w, false);
        var sigmoid = new Sigmoid();
        var aSigmoid = AddEdge(a, sigmoid, true);
        const int numberOfHiddenUnitsInLayer2 = 10;
        var initialWeights2 = initialization.Initialize(numberOfHiddenUnitsInLayer1 + 1, numberOfHiddenUnitsInLayer2, new Random(1));
        int[] weightsShape2 = [numberOfHiddenUnitsInLayer1 + 1, numberOfHiddenUnitsInLayer2];
        var weightsTensor2 = new Tensor(initialWeights2, weightsShape2);
        var w2 = new MultiplicationNode(weightsTensor2);
        var a2 = AddEdge(aSigmoid, w2, false);
        var sigmoid2 = new Sigmoid();
        var aSigmoid2 = AddEdge(a2, sigmoid2, true);
        const int numberOfClasses = 3;
        var initialWeights3 = initialization.Initialize(numberOfHiddenUnitsInLayer2 + 1, numberOfClasses, new Random(1));
        int[] weightsShape3 = [numberOfHiddenUnitsInLayer2 + 1, numberOfClasses];
        var weightsTensor3 = new Tensor(initialWeights3, weightsShape3);
        var w3 = new MultiplicationNode(weightsTensor3);
        var a3 = AddEdge(aSigmoid2, w3, false);
        var softmax = new Softmax();
        AddEdge(a3, softmax, false);
        for (var i = 0; i < parameters.GetEpoch(); i++)
        {
            foreach (var instance in trainSet)
            {
                input.SetValue(CreateInputTensor(instance));
                var calculatedClasses = ForwardCalculation();
                int[] index = [instance.GetShape()[0] - 1];
                List<int> classes = [(int) instance.GetValue(index)];
                Backpropagation(optimizer, classes);
            }
        }
    }
}