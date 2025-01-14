using Math;

namespace ComputationalGraph;

public class Softmax : IFunction {
    
    public Matrix? Calculate(Matrix? matrix) {
        var result = new Matrix(matrix.GetRow(), matrix.GetColumn());
        for (var i = 0; i < matrix.GetRow(); i++) {
            var sum = 0.0;
            for (var k = 0; k < matrix.GetColumn(); k++) {
                sum += System.Math.Exp(matrix.GetValue(i, k));
            }
            for (var k = 0; k < matrix.GetColumn(); k++) {
                result.SetValue(i, k, System.Math.Exp(matrix.GetValue(i, k)) / sum);
            }
        }
        return result;
    }

    public Matrix? Derivative(Matrix? matrix) {
        return null;
    }
}