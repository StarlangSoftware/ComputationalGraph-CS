using System.Collections.Generic;
using Math;

namespace ComputationalGraph.Function;

public class Tanh : Function {
    
    /**
     * <summary>Computes the Tanh activation for the given tensor.</summary>
     * <param name="value">The tensor whose values are to be computed.</param>
     * <returns>Tanh(x).</returns>
     */
    public Tensor Calculate(Tensor value) {
        var values = new List<double>();
        var oldValues = value.GetData();
        foreach (var oldValue in oldValues)
        {
            values.Add(System.Math.Tanh(oldValue));
        }
        return new Tensor(values, value.GetShape());
    }

    /**
     * <summary>Computes the derivative of the Tanh activation function.</summary>
     * <param name="value"> output of the Tanh(x).</param>
     * <param name="backward"> Backward tensor.</param>
     * <returns> Gradient value of the corresponding node.</returns>
     */
    public Tensor Derivative(Tensor value, Tensor backward) {
        var values = new List<double>();
        var oldValues = value.GetData();
        var backwardValues = backward.GetData();
        for (var i = 0; i < oldValues.Count; i++)
        {
            var oldValue =  oldValues[i];
            var backwardValue = backwardValues[i];
            values.Add((1 - oldValue * oldValue) * backwardValue);
        }
        return new Tensor(values, value.GetShape());
    }
}