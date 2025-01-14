using Math;

namespace ComputationalGraph;

public class Tanh : IFunction {
    
    public Matrix? Calculate(Matrix? matrix) {
        var result = new Matrix(matrix.GetRow(), matrix.GetColumn());
        for (var i = 0; i < matrix.GetRow(); i++) {
            for (var j = 0; j < matrix.GetColumn(); j++) {
                result.SetValue(i, j, System.Math.Tanh(matrix.GetValue(i, j)));
            }
        }
        return result;
    }

    public Matrix? Derivative(Matrix? matrix) {
        var result = new Matrix(matrix.GetRow(), matrix.GetColumn());
        for (var i = 0; i < matrix.GetRow(); i++) {
            for (var j = 0; j < matrix.GetColumn(); j++) {
                result.SetValue(i, j, 1 - (matrix.GetValue(i, j) * matrix.GetValue(i, j)));
            }
        }
        return result;
    }
}