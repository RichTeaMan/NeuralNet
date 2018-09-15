using RichTea.Common;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{

    /// <summary>
    /// A neural node.
    /// </summary>
    public class Node
    {
        #region Properties

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
        public double Result { get; private set; }

        #endregion

        #region Methods

        /// <summary>
        /// Constructs the node with random values for the weights and bias.
        /// </summary>
        /// <param name="inputs">The number of inputs the Node should have.</param>
        public Node(int inputs, Random random)
        {
            if (inputs == 0)
            {
                throw new ArgumentException("A Node must have at least one input");
            }

            Bias = random.NextDouble();
            Weights = Enumerable.Range(0, inputs).Select(i => random.NextDouble()).ToArray();
        }

        /// <summary>
        /// Initialises node from given weights and bias.
        /// </summary>
        ///<param name="bias">Bias.</param>
        ///<param name="weights">Weights.</param>
        public Node(double bias, IEnumerable<double> weights)
        {
            Bias = bias;
            Weights = weights.ToArray();
        }

        /// <summary>
        /// Clones another node
        /// </summary>
        /// <param name="node">Node.</param>
        public Node(Node node)
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
        public double Calculate(double[] inputs)
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

        /// <summary>
        /// Calculate.
        /// </summary>
        /// <param name="inputs">Inputs.</param>
        /// <param name="target">Target.</param>
        /// <param name="delta">Delta.</param>
        /// <returns>Result.</returns>
        public double Calculate(double[] inputs, double target, ref double delta)
        {
            double result = Calculate(inputs);
            delta = (target - result) * result * (1 - result);
            return result;
        }

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
