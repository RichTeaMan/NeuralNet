﻿using RichTea.Common;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{

    /// <summary>
    /// A neural node.
    /// </summary>
    public abstract class Node
    {
        #region Properties

        public abstract NodeType NodeType { get; }

        /// <summary>
        /// Gets input count.
        /// </summary>
        public int InputCount
        {
            get
            {
                return Weights.Count;
            }
        }

        /// <summary>
        /// Gets bias.
        /// </summary>
        public double Bias { get; }

        /// <summary>
        /// Gets weights.
        /// </summary>
        public IReadOnlyList<double> Weights { get; }

        /// <summary>
        /// Gets the result of the last calculation.
        /// </summary>
        public double Result { get; protected set; }

        #endregion

        #region Methods

        /// <summary>
        /// Constructs the node with random values for the weights and bias.
        /// </summary>
        /// <param name="inputs">The number of inputs the Node should have.</param>
        /// <param name="random">Random.</param>
        public Node(int inputs, Random random)
        {
            if (inputs <= 0)
            {
                throw new ArgumentException("A Node must have at least one input");
            }

            Bias = 0.1; // random.NextDouble();
            Weights = Enumerable.Range(0, inputs).Select(i => random.NextDouble() / 10).ToArray();
        }

        /// <summary>
        /// Initialises node from given weights and bias.
        /// </summary>
        ///<param name="bias">Bias.</param>
        ///<param name="weights">Weights.</param>
        protected Node(double bias, IEnumerable<double> weights)
        {
            Bias = bias;
            Weights = weights.ToArray();
        }

        /// <summary>
        /// Clones another node
        /// </summary>
        /// <param name="node">Node.</param>
        protected Node(Node node)
        {
            Bias = node.Bias;
            // ToArray() to ensure the same array instance is not reused.
            Weights = node.Weights.ToArray();
        }

        /// <summary>
        /// Calculates result from inputs.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <returns>Result.</returns>
        public abstract double Calculate(double[] inputs);

        /// <summary>
        /// Calculate.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <param name="target">Target.</param>
        /// <param name="delta">Delta.</param>
        /// <returns>Result.</returns>
        public virtual double Calculate(double[] inputs, double target, ref double delta)
        {
            double result = Calculate(inputs);
            delta = (target - result) * CalculateDerivative(inputs);
            return result;
        }

        /// <summary>
        /// Calculate.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <returns>Result.</returns>
        public virtual double CalculateDerivative(double[] inputs)
        {
            double result = Calculate(inputs);
            return CalculateDerivative(result);
        }

        public abstract double CalculateDerivative(double input);

        /// <summary>
        /// Creates node for serialisation.
        /// </summary>
        /// <returns></returns>
        public SerialisedNode CreateSerialisedNode()
        {
            var serialisedNode = new SerialisedNode
            {
                Bias = Bias,
                Weights = Weights.ToArray()
            };

            return serialisedNode;
        }

        #endregion

        /// <summary>
        /// Generates string representation of node.
        /// </summary>
        /// <returns></returns>
        public override string ToString()
        {
            return new ToStringBuilder<Node>(this)
                .Append(p => p.Bias)
                .Append(p => p.Weights)
                .ToString();
        }

    }
}
