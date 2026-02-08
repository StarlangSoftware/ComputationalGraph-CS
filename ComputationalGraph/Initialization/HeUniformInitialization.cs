using System;
using System.Collections.Generic;

namespace ComputationalGraph.Initialization;

public class HeUniformInitialization : Initialization
{
    /**
     * <summary>He Uniform Initialization.
     * <p>
     * This method initializes weights using a uniform distribution, which is typically
     * optimized for layers with ReLU activation functions. It helps in maintaining
     * the variance of activations throughout the network layers.
     * </p></summary>
     *
     * <param name="row">    The number of rows in the matrix (typically represents the output size / number of neurons).</param>
     * <param name="column"> The number of columns in the matrix (typically represents the input size / fan-in).</param>
     * <param name="random"> The {@link Random} instance used for generating values (allows for reproducibility).</param>
     * <returns> An {@link ArrayList} of Doubles containing the initialized weight values.</returns>
     */
    public List<double> Initialize(int row, int column, Random random)
    {
        var data = new List<double>();
        for (var i = 0; i < row; i++)
        {
            for (var j = 0; j < column; j++)
            {
                data.Add(((System.Math.Sqrt(6.0 / column) + System.Math.Sqrt(6.0 / row)) * random.NextDouble()) - System.Math.Sqrt(6.0 / row));
            }
        }
        return data;
    }
}