using System;
using System.Collections.Generic;

namespace ComputationalGraph.Initialization;

public interface Initialization
{
    List<double> Initialize(int row, int column, Random random);
}