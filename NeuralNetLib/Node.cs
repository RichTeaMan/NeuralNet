using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public class Node : INode
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
        /// <param name="Inputs">The number of inputs the Node should have.</param>
        public Node(int Inputs)
        {
            if (Inputs == 0)
                throw new ArgumentException("A NodeLayer must have at least one input");

            Random rand = new Random();

            Bias = rand.NextDouble();

            Weights = new double[Inputs];
            for (int i = 0; i < Inputs; i++)
            {
                Weights[i] = rand.NextDouble();
            }
        }

        public double Calculate(double[] Inputs)
        {
            if (this.Inputs != Inputs.Length)
                throw new ArgumentException("There is not a correct number of inputs.");


            double result = Bias;
            for (int i = 0; i < this.Inputs; i++)
            {
                result += Inputs[i] * Weights[i];
            }
            Result = 1.0 / (1.0 + Math.Exp(-result));
            return Result;
        }

        public double Calculate(double[] Inputs, double Target, ref double Delta)
        {
            double result = Calculate(Inputs);
            Delta = (Target - result) * result * (1 - result);
            return result;
        }

        #endregion
    }
}
