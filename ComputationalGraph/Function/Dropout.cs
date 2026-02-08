using System;
using System.Collections.Generic;
using Math;

namespace ComputationalGraph.Function;

public class Dropout : Function
{
    private readonly double _p;
    private readonly List<double> _mask;
    private readonly Random _random;

    public Dropout(double p, Random random)
    {
        _p = p;
        _random = random;
        _mask = [];
    }
    
    /**
     * <summary>Computes the dropout values for the given value tensor.</summary>
     * <param name="value"> The tensor whose values are to be computed.</param>
     * <returns> Output tensor.</returns>
     */
    public Tensor Calculate(Tensor value)
    {
        this._mask.Clear();
        var multiplier = 1.0 / (1 - _p);
        var values = new List<double>();
        var oldValues = value.GetData();
        foreach (var oldValue in oldValues)
        {
            var r = _random.NextDouble();
            if (r > _p)
            {
                _mask.Add(multiplier);
                values.Add(oldValue * multiplier);
            }
            else
            {
                _mask.Add(0.0);
                values.Add(0.0);
            }
        }

        return new Tensor(values, value.GetShape());
    }

    /**
     * <summary>Calculates the derivative of the dropout.</summary>
     * <param name="value"> output of the dropout function.</param>
     * <param name="backward"> Backward tensor.</param>
     * <returns> Gradient value of the corresponding node.</returns>
     */
    public Tensor Derivative(Tensor value, Tensor backward)
    {
        return backward.HadamardProduct(new Tensor(_mask, value.GetShape()));
    }
}