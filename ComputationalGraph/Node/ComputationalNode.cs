using System;
using System.Collections.Generic;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Node
{
    [Serializable]
    public class ComputationalNode
    {
        protected Tensor value;
        protected Tensor backward;
        protected readonly bool isBiased;
        protected readonly bool learnable;
        private readonly List<ComputationalNode> children;
        private readonly List<ComputationalNode> parents;

        public ComputationalNode(bool learnable, bool isBiased, Tensor value)
        {
            this.value = value;
            this.backward = null;
            this.isBiased = isBiased;
            this.learnable = learnable;
            children = new List<ComputationalNode>();
            parents = new List<ComputationalNode>();
        }

        public ComputationalNode(bool learnable, bool isBiased)
            : this(learnable, isBiased, null)
        {
        }

        public ComputationalNode()
            : this(false, false)
        {
        }

        public ComputationalNode getChild(int index)
        {
            return children[index];
        }

        public void addChild(ComputationalNode child)
        {
            children.Add(child);
        }

        public void addParent(ComputationalNode parent)
        {
            parents.Add(parent);
        }

        public void add(ComputationalNode child)
        {
            children.Add(child);
            child.addParent(this);
        }

        public ComputationalNode getParent(int index)
        {
            return parents[index];
        }

        public int childrenSize()
        {
            return children.Count;
        }

        public int parentsSize()
        {
            return parents.Count;
        }

        public bool isLearnable()
        {
            return learnable;
        }

        public override string ToString()
        {
            string details = "";

            if (value != null)
            {
                if (details.Length > 0)
                {
                    details += ", ";
                }

                details += "Value Shape: [" + value.GetShape()[0];
                for (int i = 1; i < value.GetShape().Length; i++)
                {
                    details += ", " + value.GetShape()[i];
                }

                details += "]";
            }

            if (details.Length > 0)
            {
                details += ", ";
            }

            details += "is learnable: " + learnable;
            details += ", is biased: " + isBiased;

            return "Node(" + details + ")";
        }

        public bool isBiasedNode()
        {
            return isBiased;
        }

        public Tensor getValue()
        {
            return value;
        }

        public void setValue(Tensor value)
        {
            this.value = value;
        }

        public void updateValue()
        {
            value.Add(backward);
        }

        public Tensor getBackward()
        {
            return backward;
        }

        public void setBackward(Tensor backward)
        {
            this.backward = backward;
        }
    }
}