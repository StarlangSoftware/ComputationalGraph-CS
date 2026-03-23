using System;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Node
{
    [Serializable]
    public class MultiplicationNode : ComputationalNode
    {
        private readonly bool _isHadamard;
        private readonly ComputationalNode _priorityNode;

        /**
         * <summary>Creates a multiplication node with the given learnability, bias flag, and Hadamard setting.</summary>
         *
         * <param name="learnable">Indicates whether the node is learnable.</param>
         * <param name="isBiased">Indicates whether the node is biased.</param>
         * <param name="isHadamard">Indicates whether the multiplication is Hadamard multiplication.</param>
         */
        public MultiplicationNode(bool learnable, bool isBiased, bool isHadamard)
            : base(learnable, isBiased)
        {
            _isHadamard = isHadamard;
            _priorityNode = null;
        }

        /**
         * <summary>Creates a multiplication node with the given learnability, bias flag, Hadamard setting, and priority node.</summary>
         *
         * <param name="learnable">Indicates whether the node is learnable.</param>
         * <param name="isBiased">Indicates whether the node is biased.</param>
         * <param name="isHadamard">Indicates whether the multiplication is Hadamard multiplication.</param>
         * <param name="priorityNode">Priority node of the multiplication.</param>
         */
        public MultiplicationNode(bool learnable, bool isBiased, bool isHadamard, ComputationalNode priorityNode)
            : base(learnable, isBiased)
        {
            _isHadamard = isHadamard;
            _priorityNode = priorityNode;
        }

        /**
         * <summary>Creates a multiplication node with the given learnability, bias flag, value, and Hadamard setting.</summary>
         *
         * <param name="learnable">Indicates whether the node is learnable.</param>
         * <param name="isBiased">Indicates whether the node is biased.</param>
         * <param name="value">Tensor value of the node.</param>
         * <param name="isHadamard">Indicates whether the multiplication is Hadamard multiplication.</param>
         */
        public MultiplicationNode(bool learnable, bool isBiased, Tensor value, bool isHadamard)
            : base(learnable, isBiased, value)
        {
            _isHadamard = isHadamard;
            _priorityNode = null;
        }

        /**
         * <summary>Creates a multiplication node with the given learnability and value.</summary>
         *
         * <param name="learnable">Indicates whether the node is learnable.</param>
         * <param name="value">Tensor value of the node.</param>
         */
        public MultiplicationNode(bool learnable, Tensor value)
            : base(learnable, false, value)
        {
            _priorityNode = null;
            _isHadamard = false;
        }

        /**
         * <summary>Creates a multiplication node with the given tensor value.</summary>
         *
         * <param name="value">Tensor value of the node.</param>
         */
        public MultiplicationNode(Tensor value)
            : base(true, false, value)
        {
            _priorityNode = null;
            _isHadamard = false;
        }

        /**
         * <summary>Creates a multiplication node with the given learnability and bias flag.</summary>
         *
         * <param name="learnable">Indicates whether the node is learnable.</param>
         * <param name="isBiased">Indicates whether the node is biased.</param>
         */
        public MultiplicationNode(bool learnable, bool isBiased)
            : base(learnable, isBiased)
        {
            _isHadamard = false;
            _priorityNode = null;
        }

        /**
         * <summary>Returns a string representation of the multiplication node.</summary>
         *
         * <returns>A string representation of the multiplication node.</returns>
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

            return "MultiplicationNode(" + details + ")";
        }

        /**
         * <summary>Returns whether the multiplication is Hadamard multiplication.</summary>
         *
         * <returns>True if the multiplication is Hadamard multiplication; otherwise, false.</returns>
         */
        public bool IsHadamard()
        {
            return _isHadamard;
        }

        /**
         * <summary>Returns the priority node of the multiplication node.</summary>
         *
         * <returns>The priority node if it exists; otherwise, null.</returns>
         */
        public ComputationalNode GetPriorityNode()
        {
            return _priorityNode;
        }
    }
}