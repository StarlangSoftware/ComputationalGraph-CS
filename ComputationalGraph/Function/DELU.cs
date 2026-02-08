using System.Collections.Generic;
using Math;

namespace ComputationalGraph.Function;

public class DELU : Function
{
    private readonly double _a;
    private readonly double _b;
    private readonly double _xc;

    public DELU(double a, double b, double c)
    {
        _a = a;
        _b = b;
        _xc = c;
    }

    public DELU()
    {
        _a = 1.0;
        _b = 2.0;
        _xc = 1.25643;
    }
    
    /**
     * <summary> Computes the DELU activation for the given value tensor.</summary>
     * <param name="value">The tensor whose values are to be computed.</param>
     * <returns>DELU(x).</returns>
     */
    public Tensor Calculate(Tensor value)
    {
        var values = new List<double>();
        var oldValues = value.GetData();
        foreach (var oldValue in oldValues) {
            if (oldValue > _xc)
            {
                values.Add(oldValue);
            }
            else
            {
                values.Add((System.Math.Exp(_a * oldValue) - 1) /  _b);
            }
        }
        return new Tensor(values, value.GetShape());
    }

    /**
     * <summary>Computes the derivative of the DELU activation function.</summary>
     * <param name="value">output of the DELU(x).</param>
     * <param name="backward">Backward tensor.</param>
     * <returns>Gradient value of the corresponding node.</returns>
     */
    public Tensor Derivative(Tensor value, Tensor backward)
    {
        var values = new List<double>();
        var oldValues = value.GetData();
        var backwardValues = backward.GetData();
        for (var i = 0; i < oldValues.Count; i++) {
            var oldValue = oldValues[i];
            var backwardValue = backwardValues[i];
            if (oldValue > _xc)
            {
                values.Add(backwardValue);
            }
            else
            {
                values.Add(backwardValue * ((oldValue * _b + 1) * (_a / _b)));
            }
        }
        return new Tensor(values, value.GetShape());
    }
}