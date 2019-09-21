using RichTea.Common;
using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{

    /// <summary>
    /// Hyperbolic tangent function
    /// </summary>
    public class HyperbolicTangentNode : Node
    {
        public override NodeType NodeType => NodeType.HyperbolicTangent;

        /// <summary>
        /// Constructs the node with random values for the weights and bias.
        /// </summary>
        /// <param name="inputs">The number of inputs the Node should have.</param>
        /// <param name="random">Random.</param>
        public HyperbolicTangentNode(int inputs, Random random) : base(inputs, random)
        {
            Bias = 0.05;// random.NextDouble() / 1000;
            UpdateWeights(Weights.Select(w => 0.05));// random.NextDouble() / 100));
        }

        /// <summary>
        /// Initialises node from given weights and bias.
        /// </summary>
        ///<param name="bias">Bias.</param>
        ///<param name="weights">Weights.</param>
        public HyperbolicTangentNode(double bias, IEnumerable<double> weights) : base(bias, weights) { }

        /// <summary>
        /// Clones another node
        /// </summary>
        /// <param name="node">Node.</param>
        public HyperbolicTangentNode(Node node) : base(node) { }

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
            Result = Math.Tanh(result);
            if (Result > 1.0)
            {
                Console.WriteLine("???");
            }
            return Result;
        }

        public override double CalculateDerivative(double result)
        {
            return 1.0 - Math.Pow(Math.Tanh(result), 2);
        }

        public override void UpdateBias(double updatedBias)
        {
            double adjusted = Limit(updatedBias);
            base.UpdateBias(adjusted);
        }

        public override void UpdateWeights(IEnumerable<double> updatedWeights)
        {
            base.UpdateWeights(updatedWeights.Select(w => Limit(w)));
        }

        private static double Limit(double updatedBias)
        {
            double adjusted = Math.Min(1.0, updatedBias);
            adjusted = Math.Max(-1, adjusted);
            return adjusted;
        }

    }
}
