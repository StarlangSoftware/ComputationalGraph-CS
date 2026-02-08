using System.Collections.Generic;

namespace ComputationalGraph.Node;

public class ConcatenatedNode : ComputationalNode
{
    private readonly Dictionary<ComputationalNode, int> _indexMap;
    private readonly int _dimension;
    
    public ConcatenatedNode(int dimension) : base(false, false, null, null)
    {
        _indexMap = new Dictionary<ComputationalNode, int>();
        _dimension = dimension;
    }
    
    public int GetDimension()
    {
        return _dimension;
    }

    public int GetIndex(ComputationalNode node)
    {
        return _indexMap.ContainsKey(node) ? _indexMap[node] : -1;
    }

    public void AddNode(ComputationalNode node)
    {
        _indexMap.Add(node, _indexMap.Count);
    }
}