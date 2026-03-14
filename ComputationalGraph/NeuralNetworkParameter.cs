using System;
using System.Collections.Generic;
using Classification.Parameter;
using ComputationalGraph.Initialization;
using CGOptimizer = ComputationalGraph.Optimizer.Optimizer;
using CGFunction = ComputationalGraph.Function.Function;
using CrossEntropyLossFunction = global::ComputationalGraph.Function.CrossEntropyLoss;

namespace ComputationalGraph
{
    [Serializable]
    public class NeuralNetworkParameter : Parameter
    {
        private readonly CGOptimizer optimizer;
        private readonly int epoch;
        private readonly Initialization.Initialization initialization;
        private readonly double dropout;
        private readonly CGFunction lossFunction;
        private readonly int batchSize;

        public NeuralNetworkParameter(
            int seed,
            int epoch,
            CGOptimizer optimizer,
            Initialization.Initialization initialization,
            CGFunction lossFunction,
            double dropout,
            int batchSize)
            : base(seed)
        {
            this.optimizer = optimizer;
            this.epoch = epoch;
            this.initialization = initialization;
            this.dropout = dropout;
            this.lossFunction = lossFunction;
            this.batchSize = batchSize;
        }

        public NeuralNetworkParameter(int seed, int epoch, CGOptimizer optimizer)
            : base(seed)
        {
            this.optimizer = optimizer;
            this.epoch = epoch;
            this.initialization = new RandomInitialization();
            this.dropout = 0.0;
            this.lossFunction = new CrossEntropyLossFunction();
            this.batchSize = 1;
        }

        public NeuralNetworkParameter(int seed, int epoch, CGOptimizer optimizer, CGFunction lossFunction, double dropout)
            : base(seed)
        {
            this.optimizer = optimizer;
            this.epoch = epoch;
            this.initialization = new RandomInitialization();
            this.dropout = dropout;
            this.lossFunction = lossFunction;
            this.batchSize = 1;
        }

        public CGOptimizer getOptimizer()
        {
            return optimizer;
        }

        public int getEpoch()
        {
            return epoch;
        }

        public List<double> initializeWeights(int row, int column, Random random)
        {
            return initialization.Initialize(row, column, random);
        }

        public double getDropout()
        {
            return dropout;
        }

        public CGFunction getLossFunction()
        {
            return lossFunction;
        }

        public int getBatchSize()
        {
            return batchSize;
        }
    }
}