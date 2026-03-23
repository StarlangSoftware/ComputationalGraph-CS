using System;
using System.Collections.Generic;

namespace ComputationalGraph.Initialization
{
    [Serializable]
    public class UniformXavierInitialization : Initialization
    {
        /**
         * <summary>Initializes weights using Xavier uniform initialization.</summary>
         *
         * <param name="row">The number of rows in the matrix.</param>
         * <param name="column">The number of columns in the matrix.</param>
         * <param name="random">Random object used for generating values.</param>
         * <returns>A list of initialized weight values.</returns>
         */
        public List<double> Initialize(int row, int column, Random random)
        {
            var data = new List<double>();

            for (var i = 0; i < row; i++)
            {
                for (var j = 0; j < column; j++)
                {
                    data.Add((2 * random.NextDouble() - 1) * System.Math.Sqrt(6.0 / (row + column)));
                }
            }

            return data;
        }
    }
}