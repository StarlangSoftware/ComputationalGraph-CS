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
        private readonly CGOptimizer _optimizer;
        private readonly int _epoch;
        private readonly Initialization.Initialization _initialization;
        private readonly double _dropout;
        private readonly CGFunction _lossFunction;
        private readonly int _batchSize;

        /**
         * <summary>Creates a neural network parameter object with all configuration values.</summary>
         *
         * <param name="seed">Random seed value.</param>
         * <param name="epoch">Number of training epochs.</param>
         * <param name="optimizer">Optimizer of the neural network.</param>
         * <param name="initialization">Initialization strategy of the network weights.</param>
         * <param name="lossFunction">Loss function of the neural network.</param>
         * <param name="dropout">Dropout rate.</param>
         * <param name="batchSize">Batch size used during training.</param>
         */
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
            _optimizer = optimizer;
            _epoch = epoch;
            _initialization = initialization;
            _dropout = dropout;
            _lossFunction = lossFunction;
            _batchSize = batchSize;
        }

        /**
         * <summary>Creates a neural network parameter object with default initialization, loss function, dropout, and batch size.</summary>
         *
         * <param name="seed">Random seed value.</param>
         * <param name="epoch">Number of training epochs.</param>
         * <param name="optimizer">Optimizer of the neural network.</param>
         */
        public NeuralNetworkParameter(int seed, int epoch, CGOptimizer optimizer)
            : base(seed)
        {
            _optimizer = optimizer;
            _epoch = epoch;
            _initialization = new RandomInitialization();
            _dropout = 0.0;
            _lossFunction = new CrossEntropyLossFunction();
            _batchSize = 1;
        }

        /**
         * <summary>Creates a neural network parameter object with default initialization and batch size.</summary>
         *
         * <param name="seed">Random seed value.</param>
         * <param name="epoch">Number of training epochs.</param>
         * <param name="optimizer">Optimizer of the neural network.</param>
         * <param name="lossFunction">Loss function of the neural network.</param>
         * <param name="dropout">Dropout rate.</param>
         */
        public NeuralNetworkParameter(int seed, int epoch, CGOptimizer optimizer, CGFunction lossFunction, double dropout)
            : base(seed)
        {
            _optimizer = optimizer;
            _epoch = epoch;
            _initialization = new RandomInitialization();
            _dropout = dropout;
            _lossFunction = lossFunction;
            _batchSize = 1;
        }

        /**
         * <summary>Returns the optimizer of the neural network.</summary>
         *
         * <returns>The optimizer of the neural network.</returns>
         */
        public CGOptimizer GetOptimizer()
        {
            return _optimizer;
        }

        /**
         * <summary>Returns the number of training epochs.</summary>
         *
         * <returns>The number of training epochs.</returns>
         */
        public int GetEpoch()
        {
            return _epoch;
        }

        /**
         * <summary>Initializes the weights for the given matrix dimensions.</summary>
         *
         * <param name="row">Number of rows.</param>
         * <param name="column">Number of columns.</param>
         * <param name="random">Random object used for initialization.</param>
         * <returns>A list of initialized weight values.</returns>
         */
        public List<double> InitializeWeights(int row, int column, Random random)
        {
            return _initialization.Initialize(row, column, random);
        }

        /**
         * <summary>Returns the dropout rate.</summary>
         *
         * <returns>The dropout rate.</returns>
         */
        public double GetDropout()
        {
            return _dropout;
        }

        /**
         * <summary>Returns the loss function of the neural network.</summary>
         *
         * <returns>The loss function of the neural network.</returns>
         */
        public CGFunction GetLossFunction()
        {
            return _lossFunction;
        }

        /**
         * <summary>Returns the batch size.</summary>
         *
         * <returns>The batch size.</returns>
         */
        public int GetBatchSize()
        {
            return _batchSize;
        }
    }
}