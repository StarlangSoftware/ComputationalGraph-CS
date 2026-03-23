using System;
using System.Collections.Generic;

namespace ComputationalGraph.Initialization
{
    [Serializable]
    public class RandomInitialization : Initialization
    {
        /**
         * <summary>Initializes weights with small uniformly distributed random values.</summary>
         *
         * <param name="row">The number of rows in the matrix.</param>
         * <param name="column">The number of columns in the matrix.</param>
         * <param name="random">Random object used for generating values.</param>
         * <returns>A list of initialized weight values.</returns>
         */
        public List<double> Initialize(int row, int column, Random random)
        {
            var data = new List<double>();

            for (var i = 0; i < row * column; i++)
            {
                data.Add(-0.01 + (0.02 * random.NextDouble()));
            }

            return data;
        }
    }
}