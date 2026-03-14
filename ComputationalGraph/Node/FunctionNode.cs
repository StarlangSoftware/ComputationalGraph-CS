using System;
using CGFunction = ComputationalGraph.Function.Function;

namespace ComputationalGraph.Node
{
    [Serializable]
    public class FunctionNode : ComputationalNode
    {
        private readonly CGFunction function;

        public FunctionNode(bool isBiased, CGFunction function)
            : base(false, isBiased)
        {
            this.function = function;
        }

        public FunctionNode(bool learnable, bool isBiased, CGFunction function)
            : base(learnable, isBiased)
        {
            this.function = function;
        }

        public override string ToString()
        {
            string details = "";

            if (function != null)
            {
                details += "Function: " + function;
            }

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

            details += ", is biased: " + isBiased;
            return "FunctionNode(" + details + ")";
        }

        public CGFunction getFunction()
        {
            return function;
        }
    }
}