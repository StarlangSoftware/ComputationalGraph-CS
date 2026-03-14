using System;
using System.Collections.Generic;

namespace ComputationalGraph.Node
{
    [Serializable]
    public class ConcatenatedNode : ComputationalNode
    {
        private readonly Dictionary<ComputationalNode, int> indexMap;
        private readonly int dimension;

        public ConcatenatedNode(int dimension)
            : base(false, false)
        {
            this.indexMap = new Dictionary<ComputationalNode, int>();
            this.dimension = dimension;
        }

        public int getDimension()
        {
            return dimension;
        }

        public int getIndex(ComputationalNode node)
        {
            return indexMap[node];
        }

        public void addNode(ComputationalNode node)
        {
            indexMap[node] = indexMap.Count;
        }
    }
}