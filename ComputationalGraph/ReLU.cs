using Math;

namespace ComputationalGraph;

public class ReLU : IFunction {
    
    public Matrix? Calculate(Matrix? matrix) {
        var result = new Matrix(matrix.GetRow(), matrix.GetColumn());
        for (var i = 0; i < matrix.GetRow(); i++) {
            for (var j = 0; j < matrix.GetColumn(); j++) {
                if (matrix.GetValue(i, j) > 0) {
                    result.SetValue(i, j, matrix.GetValue(i, j));
                } else {
                    result.SetValue(i, j, 0.0);
                }
            }
        }
        return result;
    }

    public Matrix? Derivative(Matrix? matrix) {
        var result = new Matrix(matrix.GetRow(), matrix.GetColumn());
        for (var i = 0; i < matrix.GetRow(); i++) {
            for (var j = 0; j < matrix.GetColumn(); j++) {
                if (matrix.GetValue(i, j) != 0) {
                    result.SetValue(i, j, 1.0);
                } else {
                    result.SetValue(i, j, 0.0);
                }
            }
        }
        return result;
    }
}