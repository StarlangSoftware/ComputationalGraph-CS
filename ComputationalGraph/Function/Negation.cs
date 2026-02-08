using System.Collections.Generic;
using Math;

namespace ComputationalGraph.Function;

public class Negation : Function
{
    /**
     * <summary>Negates the values of the given tensor.</summary>
     * <param name="value"> The tensor whose values are to be negated.</param>
     * <returns> The negated tensor.</returns>
     */
    public Tensor Calculate(Tensor value)
    {
        var values = new List<double>();
        var oldValues = value.GetData();
        foreach (var oldValue in oldValues) {
            values.Add(-oldValue);
        }
        return new Tensor(values, value.GetShape());
    }

    /**
     * <summary>Calculates the derivative of the Negation function.</summary>
     * <param name="value"> output of the Negation function.</param>
     * <param name="backward"> Backward tensor.</param>
     * <returns> Gradient value of the corresponding node.</returns>
     */
    public Tensor Derivative(Tensor value, Tensor backward)
    {
        var values = new List<double>();
        var backwardValues = backward.GetData();
        foreach (var backwardValue in backwardValues) {
            values.Add(-backwardValue);
        }
        return new Tensor(values, value.GetShape());
    }
}