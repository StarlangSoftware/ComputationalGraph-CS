using System;
using System.Collections.Generic;

namespace ComputationalGraph.Initialization;

/**
 * <summary>Defines the initialization behavior for creating matrix values.</summary>
 */
public interface Initialization
{
    /**
     * <summary>Initializes a list of values for the given row and column dimensions.</summary>
     *
     * <param name="row">Number of rows.</param>
     * <param name="column">Number of columns.</param>
     * <param name="random">Random object used during initialization.</param>
     * <returns>A list of initialized double values.</returns>
     */
    List<double> Initialize(int row, int column, Random random);
}