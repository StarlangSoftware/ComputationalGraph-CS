using ComputationalGraph;
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
}