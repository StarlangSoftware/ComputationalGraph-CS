using System;
using CGFunction = ComputationalGraph.Function.Function;

namespace ComputationalGraph.Node
{
    [Serializable]
    public class FunctionNode : ComputationalNode
    {
        private readonly CGFunction _function;

        /**
         * <summary>Creates a function node with the given bias flag and function.</summary>
         *
         * <param name="isBiased">Indicates whether the node is biased.</param>
         * <param name="function">Function stored in the node.</param>
         */
        public FunctionNode(bool isBiased, CGFunction function)
            : base(false, isBiased)
        {
            _function = function;
        }

        /**
         * <summary>Creates a function node with the given learnability, bias flag, and function.</summary>
         *
         * <param name="learnable">Indicates whether the node is learnable.</param>
         * <param name="isBiased">Indicates whether the node is biased.</param>
         * <param name="function">Function stored in the node.</param>
         */
        public FunctionNode(bool learnable, bool isBiased, CGFunction function)
            : base(learnable, isBiased)
        {
            _function = function;
        }

        /**
         * <summary>Returns a string representation of the function node.</summary>
         *
         * <returns>A string representation of the function node.</returns>
         */
        public override string ToString()
        {
            var details = "";

            if (_function != null)
            {
                details += "Function: " + _function;
            }

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

            details += ", is biased: " + IsBiased;
            return "FunctionNode(" + details + ")";
        }

        /**
         * <summary>Returns the function of the node.</summary>
         *
         * <returns>The function of the node.</returns>
         */
        public CGFunction GetFunction()
        {
            return _function;
        }
    }
}