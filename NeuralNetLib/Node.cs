using RichTea.Common;
using RichTea.NeuralNetLib.Serialisation;
using System;
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
                return Weights.Length;
            }
        }

        /// <summary>
        /// Gets or sets bias.
        /// </summary>
        public double Bias { get; set; }

        /// <summary>
        /// Gets or sets weights.
        /// </summary>
        public double[] Weights { get; set; }

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
        public Node(int inputs)
        {
            if (inputs == 0)
            {
                throw new ArgumentException("A NodeLayer must have at least one input");
            }

            Weights = new double[inputs];
        }

        /// <summary>
        /// Seeds weights randomly.
        /// </summary>
        /// <param name="random">Random.</param>
        public void SeedWeights(Random random)
        {
            Bias = random.NextDouble();

            for (int i = 0; i < InputCount; i++)
            {
                Weights[i] = random.NextDouble();
            }
        }

        /// <summary>
        /// Seed weights from another node.
        /// </summary>
        /// <param name="node">Node.</param>
        public void SeedWeights(Node node)
        {
            if (node.InputCount != InputCount)
            {
                throw new ArgumentException("Node has incorrect number of inputs.");
            }

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
