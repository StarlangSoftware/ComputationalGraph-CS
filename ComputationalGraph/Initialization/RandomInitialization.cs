using System;
using System.Collections.Generic;

namespace ComputationalGraph.Initialization
{
    [Serializable]
    public class RandomInitialization : Initialization
    {
        /**
         * <summary>Random Uniform Initialization.
         * <p>
         * This method initializes the weights with small random values uniformly distributed
         * between -0.01 and 0.01. This is a basic initialization strategy used to break
         * symmetry between neurons.
         * </p></summary>
         *
         * <param name="row">    The number of rows in the matrix.</param>
         * <param name="column"> The number of columns in the matrix.</param>
         * <param name="random"> The {@link Random} instance used for generating values.</param>
         * <returns> An {@link ArrayList} containing the initialized weight values.</returns>
         */
        public List<double> Initialize(int row, int column, Random random)
        {
            List<double> data = new List<double>();

            for (int i = 0; i < row * column; i++)
            {
                data.Add(-0.01 + (0.02 * random.NextDouble()));
            }

            return data;
        }
    }
}