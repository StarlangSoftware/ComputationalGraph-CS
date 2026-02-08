using System.Collections.Generic;
using Math;

namespace ComputationalGraph.Function;

public class Softmax : Function {
    
    /**
     * <summary>Computes the Softmax activation for the given tensor.</summary>
     * <param name="tensor">The tensor whose values are to be computed.</param>
     * <returns>Softmax(x).</returns>
     */
    public Tensor Calculate(Tensor tensor) {
        var values = new List<double>();
        var oldValues = tensor.GetData();
        var lastDimensionSize = tensor.GetShape()[tensor.GetShape().Length - 1];
        var sum = 0.0;
        var sumList = new List<double>();
        for (var i =  0; i < oldValues.Count; i++) {
            sum += System.Math.Exp(oldValues[i]);
            if ((i + 1) % lastDimensionSize == 0)
            {
                sumList.Add(sum);
                sum = 0.0;
            }
        }
        for (int i = 0; i < oldValues.Count; i++)
        {
            values.Add(System.Math.Exp(oldValues[i]) / sumList[i / lastDimensionSize]);
        }
        return new Tensor(values, tensor.GetShape());
    }

    /**
     * <summary>Computes the derivative of the Softmax activation function.</summary>
     * <param name="value">output of the Softmax(x).</param>
     * <param name="backward">Backward tensor.</param>
     * <returns>Gradient value of the corresponding node.</returns>
     */
    public Tensor Derivative(Tensor value, Tensor backward)
    {
        var lastDimensionSize = value.GetShape()[value.GetShape().Length - 1];
        var values = new List<double>();
        var oldValuesTensor = value.GetData();
        var oldValuesBackward = backward.GetData();
        var total = 0.0;
        for (var i = 0; i < oldValuesTensor.Count; i++)
        {
            total += oldValuesTensor[i] * oldValuesBackward[i];
            if ((i + 1) % lastDimensionSize == 0)
            {
                var startIndex = i / lastDimensionSize;
                for (var j = 0; j < lastDimensionSize; j++)
                {
                    values.Add(oldValuesBackward[startIndex * lastDimensionSize + j] - total);
                }
                total = 0.0;
            }
        }
        return value.HadamardProduct(new Tensor(values, value.GetShape()));
    }
}