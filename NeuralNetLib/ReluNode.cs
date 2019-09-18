using RichTea.Common;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{

    /// <summary>
    /// Rectified Linear Activation Function
    /// </summary>
    public class ReluNode : Node
    {
        public override NodeType NodeType => NodeType.Relu;

        /// <summary>
        /// Constructs the node with random values for the weights and bias.
        /// </summary>
        /// <param name="inputs">The number of inputs the Node should have.</param>
        /// <param name="random">Random.</param>
        public ReluNode(int inputs, Random random) : base(inputs, random) { }

        /// <summary>
        /// Initialises node from given weights and bias.
        /// </summary>
        ///<param name="bias">Bias.</param>
        ///<param name="weights">Weights.</param>
        public ReluNode(double bias, IEnumerable<double> weights) : base(bias, weights) { }

        /// <summary>
        /// Clones another node
        /// </summary>
        /// <param name="node">Node.</param>
        public ReluNode(Node node) : base(node) { }

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
            for (int i = 0; i < InputCount; i++)
            {
                result += inputs[i] * Weights[i];
            }
            if (result > 0)
            {
                Result = result;
            }
            else
            {
                Result = 0;
            }
            return Result;
        }

        /// <summary>
        /// Calculate.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <param name="target">Target.</param>
        /// <param name="delta">Delta.</param>
        /// <returns>Result.</returns>
        public override double CalculateDerivative(double[] inputs)
        {
            double result = Calculate(inputs);
            int derivative = 0;
            if (result >= 0)
            {
                derivative = 1;
            }
            return derivative;
        }

        public override double LastDerivative()
        {
            int derivative = 0;
            if (Result >= 0)
            {
                derivative = 1;
            }
            return derivative;
        }

    }
}
