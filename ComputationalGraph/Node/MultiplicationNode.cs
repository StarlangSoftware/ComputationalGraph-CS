using System;
using Tensor = Math.Tensor;

namespace ComputationalGraph.Node
{
    [Serializable]
    public class MultiplicationNode : ComputationalNode
    {
        private readonly bool _isHadamard;
        private readonly ComputationalNode priorityNode;

        public MultiplicationNode(bool learnable, bool isBiased, bool isHadamard)
            : base(learnable, isBiased)
        {
            this._isHadamard = isHadamard;
            this.priorityNode = null;
        }

        public MultiplicationNode(bool learnable, bool isBiased, bool isHadamard, ComputationalNode priorityNode)
            : base(learnable, isBiased)
        {
            this._isHadamard = isHadamard;
            this.priorityNode = priorityNode;
        }

        public MultiplicationNode(bool learnable, bool isBiased, Tensor value, bool isHadamard)
            : base(learnable, isBiased)
        {
            this._isHadamard = isHadamard;
            this.value = value;
            this.priorityNode = null;
        }

        public MultiplicationNode(bool learnable, Tensor value)
            : base(learnable, false, value)
        {
            this.value = value;
            this.priorityNode = null;
            this._isHadamard = false;
        }

        public MultiplicationNode(Tensor value)
            : base(true, false)
        {
            this.value = value;
            this.priorityNode = null;
            this._isHadamard = false;
        }

        public MultiplicationNode(bool learnable, bool isBiased)
            : base(learnable, isBiased)
        {
            this._isHadamard = false;
            this.priorityNode = null;
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

            return "MultiplicationNode(" + details + ")";
        }

        public bool isHadamard()
        {
            return _isHadamard;
        }

        public ComputationalNode getPriorityNode()
        {
            return priorityNode;
        }
    }
}