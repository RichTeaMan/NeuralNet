using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib
{
    public class Node
    {
        #region Properties

        public int Inputs
        {
            get
            {
                return Weights.Length;
            }
        }

        public double Bias { get; set; }
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

        public void SeedWeights(Random random)
        {
            Bias = random.NextDouble();

            for (int i = 0; i < Inputs; i++)
            {
                Weights[i] = random.NextDouble();
            }
        }

        public void SeedWeights(Node node)
        {
            if (node.Inputs != Inputs)
            {
                throw new ArgumentException("Node has incorrect number of inputs.");
            }

            Bias = node.Bias;
            // ToArray() to ensure the same array instance is not reused.
            Weights = node.Weights.ToArray();
        }

        public double Calculate(double[] inputs)
        {
            if (Inputs != inputs.Length)
            {
                throw new ArgumentException("There is not a correct number of inputs.");
            }

            double result = Bias;
            for (int i = 0; i < this.Inputs; i++)
            {
                result += inputs[i] * Weights[i];
            }
            Result = 1.0 / (1.0 + Math.Exp(-result));
            return Result;
        }

        public double Calculate(double[] inputs, double target, ref double delta)
        {
            double result = Calculate(inputs);
            delta = (target - result) * result * (1 - result);
            return result;
        }

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
    }
}
