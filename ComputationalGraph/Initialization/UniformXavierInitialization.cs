using System;
using System.Collections.Generic;

namespace ComputationalGraph.Initialization
{
    [Serializable]
    public class UniformXavierInitialization : Initialization
    {
        /**
         * <summary>Xavier Uniform Initialization.
         * <p>
         * This method initializes weights using a uniform distribution within the range
         * [-limit, limit], where the limit is sqrt(6 / (fan_in + fan_out)).
         * This strategy is designed to keep the scale of the gradients roughly the same
         * in all layers and is commonly used with Sigmoid or Tanh activation functions.
         * </p></summary>
         *
         * <param name="row">    The number of rows in the matrix (typically represents fan-out / output size).</param>
         * <param name="column"> The number of columns in the matrix (typically represents fan-in / input size).</param>
         * <param name="random"> The {@link Random} instance used for generating values.</param>
         * <returns> An {@link ArrayList} containing the initialized weight values.</returns>
         */
        public List<double> Initialize(int row, int column, Random random)
        {
            List<double> data = new List<double>();

            for (int i = 0; i < row; i++)
            {
                for (int j = 0; j < column; j++)
                {
                    data.Add((2 * random.NextDouble() - 1) * System.Math.Sqrt(6.0 / (row + column)));
                }
            }

            return data;
        }
    }
}