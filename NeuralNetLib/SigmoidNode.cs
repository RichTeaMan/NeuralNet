﻿using RichTea.Common;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{

    /// <summary>
    /// A sigmoid node, activated via an exponential function.
    /// </summary>
    public class SigmoidNode : Node
    {
        public override NodeType NodeType => NodeType.Sigmoid;

        /// <summary>
        /// Constructs the node with random values for the weights and bias.
        /// </summary>
        /// <param name="inputs">The number of inputs the Node should have.</param>
        /// <param name="random">Random.</param>
        public SigmoidNode(int inputs, Random random) : base(inputs, random) { }

        /// <summary>
        /// Initialises node from given weights and bias.
        /// </summary>
        ///<param name="bias">Bias.</param>
        ///<param name="weights">Weights.</param>
        public SigmoidNode(double bias, IEnumerable<double> weights) : base(bias, weights) { }

        /// <summary>
        /// Clones another node
        /// </summary>
        /// <param name="node">Node.</param>
        public SigmoidNode(Node node) : base(node) { }

        /// <summary>
        /// Calculates result from inputs.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <returns>Result.</returns>
        public override double Calculate(double[] inputs)
        {
            if (InputCount != inputs.Length)
            {
                throw new ArgumentException("There is not a correct number of inputs.");
            }

            double result = Bias;
            for (int i = 0; i < this.InputCount; i++)
            {
                result += inputs[i] * Weights[i];
            }
            Result = 1.0 / (1.0 + Math.Exp(-result));
            return Result;
        }

        public override double CalculateDerivative(double result)
        {
            return result * (1 - result);
        }
    }
}
