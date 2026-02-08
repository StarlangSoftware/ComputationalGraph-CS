using ComputationalGraph;
using ComputationalGraph.Optimizer;
using Math;
namespace Test;

public class Test
{
    [Test]
    public void LinearPerceptronSingleUnitTest()
    {
        var trainSet = new List<Tensor>();
        List<double> data1 = [1.0, 1.0];
        int[] shape1 = [2];
        var dataTensor = new Tensor(data1,  shape1);
        trainSet.Add(dataTensor);
        var graph = new LinearPerceptronSingleUnit();
        graph.Train(trainSet, new NeuralNetworkParameter(1, 1 , null));
    }
    
    [Test]
    public void LinearPerceptronTest()
    {
        var trainSet = new List<Tensor>();
        var testSet = new List<Tensor>();
        var graph = new LinearPerceptron();
        graph.CreateIrisDataSet(trainSet, testSet);
        graph.Train(trainSet, new NeuralNetworkParameter(1, 10 , new StochasticGradientDescent(0.1, 0.99)));
        var performance = graph.Test(testSet);
    }
    
    [Test]
    public void MultiLayerPerceptronTest()
    {
        var trainSet = new List<Tensor>();
        var testSet = new List<Tensor>();
        var graph = new MultiLayerPerceptron();
        graph.CreateIrisDataSet(trainSet, testSet);
        graph.Train(trainSet, new NeuralNetworkParameter(1, 10 , new StochasticGradientDescent(0.1, 0.99)));
        var performance = graph.Test(testSet);
    }
}