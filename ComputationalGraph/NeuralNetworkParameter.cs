using Parameter = Classification.Parameter.Parameter;
using ComputationalGraph.Initialization;

namespace ComputationalGraph;

public class NeuralNetworkParameter : Parameter
{
    private readonly Optimizer.Optimizer _optimizer;
    private readonly int _epoch;
    private readonly Initialization.Initialization _initialization;
    private readonly double _dropout;
    
    public NeuralNetworkParameter(int seed, int epoch, Optimizer.Optimizer optimizer, Initialization.Initialization initialization, double dropout = 0.0) : base(seed)
    {
        _optimizer = optimizer;
        _epoch = epoch;
        _initialization = initialization;
        _dropout = dropout;
    }
    
    public NeuralNetworkParameter(int seed, int epoch, Optimizer.Optimizer optimizer) : base(seed)
    {
        _optimizer = optimizer;
        _epoch = epoch;
        _initialization = new RandomInitialization();
        _dropout = 0.0;
    }

    public NeuralNetworkParameter(int seed, int epoch, Optimizer.Optimizer optimizer, double dropout) : base(seed)
    {
        _optimizer = optimizer;
        _epoch = epoch;
        _initialization = new RandomInitialization();
        _dropout = dropout;
    }
    
    public Optimizer.Optimizer GetOptimizer()
    {
        return _optimizer;
    }

    public int GetEpoch()
    {
        return _epoch;
    }

    public Initialization.Initialization GetInitialization()
    {
        return _initialization;
    }

    public double GetDropout()
    {
        return _dropout;
    }

}