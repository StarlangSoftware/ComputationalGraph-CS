using Math;

namespace ComputationalGraph;

public class Sigmoid : IFunction {
    
    public Matrix? Calculate(Matrix? matrix) {
        var result = new Matrix(matrix.GetRow(), matrix.GetColumn());
        for (var i = 0; i < matrix.GetRow(); i++) {
            for (var j = 0; j < matrix.GetColumn(); j++) {
                result.SetValue(i, j, (double)1.0F / ((double)1.0F + System.Math.Exp(-(Double)matrix.GetValue(i, j))));
            }
        }
        return result;
    }

    public Matrix? Derivative(Matrix? matrix) {
        var result = new Matrix(matrix.GetRow(), matrix.GetColumn());
        for (var i = 0; i < matrix.GetRow(); i++) {
            for (var j = 0; j < matrix.GetColumn(); j++) {
                result.SetValue(i, j, matrix.GetValue(i, j) * (1.0F - matrix.GetValue(i, j)));
            }
        }
        return result;
    }
}