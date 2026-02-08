using System.Collections.Generic;
using Math;

namespace ComputationalGraph.Function;

public class Sigmoid : Function {
    
    /**
     * <summary>Computes the Sigmoid activation for the given tensor.</summary>
     * <param name="value">The tensor whose values are to be computed.</param>
     * <returns>Sigmoid(x).</returns>
     */
    public Tensor Calculate(Tensor value) {
        var values = new List<double>();
        var tensorValues = value.GetData();
        foreach (var val in tensorValues) {
            var sigmoid = 1.0 / (1.0 + System.Math.Exp(-val));
            values.Add(sigmoid);
        }
        return new Tensor(values, value.GetShape());
    }

    /**
     * <summary>Computes the derivative of the Sigmoid activation function.</summary>
     * <param name="value">output of the Sigmoid(x).</param>
     * <param name="backward">Backward tensor.</param>
     * <returns>Gradient value of the corresponding node.</returns>
     */
    public Tensor Derivative(Tensor value, Tensor backward) {
        var values = new List<double>();
        var tensorValues = value.GetData();
        var backwardValues = backward.GetData();
        for (var i = 0; i < tensorValues.Count; i++) {
            var val = tensorValues[i];
            var derivative = val * (1 - val);
            var backwardValue = backwardValues[i];
            values.Add(derivative *  backwardValue);
        }
        return new Tensor(values, value.GetShape());
    }
}