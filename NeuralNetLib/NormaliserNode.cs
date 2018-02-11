using System;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// A node for normalising input. All output from this node is guaranteed to be between 0 and 1.
    /// </summary>
    public class NormaliserNode
    {
        /// <summary>
        /// Gets the lowest value this node has so far encountered.
        /// </summary>
        public double Minima { get; private set; } = 0;

        /// <summary>
        /// Gets the greatest value this node has so far encountered.
        /// </summary>
        public double Maxima { get; private set; } = 0;

        /// <summary>
        /// Calculates a normalised output based in the given input.
        /// </summary>
        /// <param name="input">Input.</param>
        /// <returns>Normalised output.</returns>
        public double Calculate(double input)
        {
            if (input < Minima)
            {
                Minima = input;
            }
            else if (input > Maxima)
            {
                Maxima = input;
            }

            double result = input / (Maxima - Minima);
            return result;
        }
    }
}
