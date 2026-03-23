using System;
using System.Collections.Generic;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Node
{
    [Serializable]
    public class ComputationalNode
    {
        protected Tensor Value;
        protected Tensor Backward;
        protected readonly bool IsBiased;
        protected readonly bool Learnable;
        private readonly List<ComputationalNode> _children;
        private readonly List<ComputationalNode> _parents;

        /**
         * <summary>Creates a computational node with the given learnability, bias flag, and value.</summary>
         *
         * <param name="learnable">Indicates whether the node is learnable.</param>
         * <param name="isBiased">Indicates whether the node is biased.</param>
         * <param name="value">Tensor value of the node.</param>
         */
        public ComputationalNode(bool learnable, bool isBiased, Tensor value)
        {
            Value = value;
            Backward = null;
            IsBiased = isBiased;
            Learnable = learnable;
            _children = new List<ComputationalNode>();
            _parents = new List<ComputationalNode>();
        }

        /**
         * <summary>Creates a computational node with the given learnability and bias flag.</summary>
         *
         * <param name="learnable">Indicates whether the node is learnable.</param>
         * <param name="isBiased">Indicates whether the node is biased.</param>
         */
        public ComputationalNode(bool learnable, bool isBiased)
            : this(learnable, isBiased, null)
        {
        }

        /**
         * <summary>Creates a default computational node.</summary>
         */
        public ComputationalNode()
            : this(false, false)
        {
        }

        /**
         * <summary>Returns the child node at the specified index.</summary>
         *
         * <param name="index">Index of the child node.</param>
         * <returns>The child node at the given index.</returns>
         */
        public ComputationalNode GetChild(int index)
        {
            return _children[index];
        }

        /**
         * <summary>Adds a child node to this node.</summary>
         *
         * <param name="child">Child node to be added.</param>
         */
        public void AddChild(ComputationalNode child)
        {
            _children.Add(child);
        }

        /**
         * <summary>Adds a parent node to this node.</summary>
         *
         * <param name="parent">Parent node to be added.</param>
         */
        public void AddParent(ComputationalNode parent)
        {
            _parents.Add(parent);
        }

        /**
         * <summary>Adds the given node as a child and sets this node as its parent.</summary>
         *
         * <param name="child">Child node to be connected.</param>
         */
        public void Add(ComputationalNode child)
        {
            _children.Add(child);
            child.AddParent(this);
        }

        /**
         * <summary>Returns the parent node at the specified index.</summary>
         *
         * <param name="index">Index of the parent node.</param>
         * <returns>The parent node at the given index.</returns>
         */
        public ComputationalNode GetParent(int index)
        {
            return _parents[index];
        }

        /**
         * <summary>Returns the number of child nodes.</summary>
         *
         * <returns>The number of child nodes.</returns>
         */
        public int ChildrenSize()
        {
            return _children.Count;
        }

        /**
         * <summary>Returns the number of parent nodes.</summary>
         *
         * <returns>The number of parent nodes.</returns>
         */
        public int ParentsSize()
        {
            return _parents.Count;
        }

        /**
         * <summary>Returns whether the node is learnable.</summary>
         *
         * <returns>True if the node is learnable; otherwise, false.</returns>
         */
        public bool IsLearnable()
        {
            return Learnable;
        }

        /**
         * <summary>Returns a string representation of the node.</summary>
         *
         * <returns>A string representation of the node.</returns>
         */
        public override string ToString()
        {
            var details = "";

            if (Value != null)
            {
                var shape = Value.GetShape();

                if (details.Length > 0)
                {
                    details += ", ";
                }

                details += "Value Shape: [" + shape[0];
                for (var i = 1; i < shape.Length; i++)
                {
                    details += ", " + shape[i];
                }

                details += "]";
            }

            if (details.Length > 0)
            {
                details += ", ";
            }

            details += "is learnable: " + Learnable;
            details += ", is biased: " + IsBiased;

            return "Node(" + details + ")";
        }

        /**
         * <summary>Returns whether the node is biased.</summary>
         *
         * <returns>True if the node is biased; otherwise, false.</returns>
         */
        public bool IsBiasedNode()
        {
            return IsBiased;
        }

        /**
         * <summary>Returns the tensor value of the node.</summary>
         *
         * <returns>The tensor value of the node.</returns>
         */
        public Tensor GetValue()
        {
            return Value;
        }

        /**
         * <summary>Sets the tensor value of the node.</summary>
         *
         * <param name="value">New tensor value.</param>
         */
        public void SetValue(Tensor value)
        {
            Value = value;
        }

        /**
         * <summary>Updates the tensor value using the backward tensor.</summary>
         */
        public void UpdateValue()
        {
            Value.Add(Backward);
        }

        /**
         * <summary>Returns the backward tensor of the node.</summary>
         *
         * <returns>The backward tensor of the node.</returns>
         */
        public Tensor GetBackward()
        {
            return Backward;
        }

        /**
         * <summary>Sets the backward tensor of the node.</summary>
         *
         * <param name="backward">New backward tensor.</param>
         */
        public void SetBackward(Tensor backward)
        {
            Backward = backward;
        }
    }
}