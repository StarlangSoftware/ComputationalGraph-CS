using Classification.Performance;
using ComputationalGraph;
using ComputationalGraph.Function;
using ComputationalGraph.Node;
using ComputationalGraph.Optimizer;
using Math;

namespace Test;

public class LinearPerceptronSingleUnit : NeuralNetwork
{
    protected override List<int> GetClassLabels(ComputationalNode outputNode)
    {
        List<int> labels = [0];
        return labels;
    }

    public override void Train(List<Tensor> trainSet, NeuralNetworkParameter parameters)
    {
        var optimizer = new StochasticGradientDescent(0.1, 0.99);
        var input = new MultiplicationNode(false, true);
        InputNodes.Add(input);
        List<double> initialWeights = [1.0, 1.0, 1.0, 1.0];
        int[] weightsShape = [2, 2];
        var weightsTensor = new Tensor(initialWeights, weightsShape);
        var w = new MultiplicationNode(weightsTensor);
        var a = AddEdge(input, w, false);
        var softmax = new Softmax();
        var outputNode = AddEdge(a, softmax, false);
        var dataTensor = trainSet[0];
        var input1 = CreateInputTensor(dataTensor);
        input.SetValue(input1);
        var calculatedClassLabels = ForwardCalculation(false);
        List<int> classes = [1];
        Backpropagation(optimizer, classes);
    }

    public override ClassificationPerformance Test(List<Tensor> testSet)
    {
        return new ClassificationPerformance(1.0);
    }
}