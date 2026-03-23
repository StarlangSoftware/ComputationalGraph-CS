using System;
using System.Collections.Generic;

namespace ComputationalGraph.Node
{
    [Serializable]
    public class ConcatenatedNode : ComputationalNode
    {
        private readonly Dictionary<ComputationalNode, int> _indexMap;
        private readonly int _dimension;

        /**
         * <summary>Creates a concatenated node with the given dimension.</summary>
         *
         * <param name="dimension">Concatenation dimension of the node.</param>
         */
        public ConcatenatedNode(int dimension)
            : base(false, false)
        {
            _indexMap = new Dictionary<ComputationalNode, int>();
            _dimension = dimension;
        }

        /**
         * <summary>Returns the concatenation dimension of the node.</summary>
         *
         * <returns>The concatenation dimension of the node.</returns>
         */
        public int GetDimension()
        {
            return _dimension;
        }

        /**
         * <summary>Returns the index of the given computational node.</summary>
         *
         * <param name="node">Node whose index will be returned.</param>
         * <returns>The index of the given node.</returns>
         */
        public int GetIndex(ComputationalNode node)
        {
            return _indexMap[node];
        }

        /**
         * <summary>Adds a computational node to the index map.</summary>
         *
         * <param name="node">Node to be added.</param>
         */
        public void AddNode(ComputationalNode node)
        {
            _indexMap[node] = _indexMap.Count;
        }
    }
}