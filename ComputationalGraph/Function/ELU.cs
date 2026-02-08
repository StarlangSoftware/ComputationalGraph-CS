using System.Collections.Generic;
using Math;

namespace ComputationalGraph.Function;

public class ELU : Function
{
    private readonly double _a;

    public ELU(double a)
    {
        _a = a;
    }

    public ELU()
    {
        _a = 1.0;
    }
    
    /**
     * <summary> Computes the ELU activation for the given value tensor.</summary>
     * <param name="value">The tensor whose values are to be computed.</param>
     * <returns>ELU(x).</returns>
     */
    public Tensor Calculate(Tensor value)
    {
        var values = new List<double>();
        var oldValues = value.GetData();
        foreach (var oldValue in oldValues) {
            if (oldValue < 0)
            {
                values.Add(_a * (System.Math.Exp(oldValue) - 1));
            }
            else
            {
                values.Add(oldValue);
            }
        }
        return new Tensor(values, value.GetShape());
    }

    /**
     * <summary>Computes the derivative of the ELU activation function.</summary>
     * <param name="value">output of the ELU(x).</param>
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
            if (oldValue < 0)
            {
                values.Add((oldValue + _a) * backwardValue);
            }
            else
            {
                values.Add(backwardValue);
            }
        }
        return new Tensor(values, value.GetShape());
    }}