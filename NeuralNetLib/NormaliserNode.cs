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
        public double Minima { get; private set; } = double.NaN;

        /// <summary>
        /// Gets the greatest value this node has so far encountered.
        /// </summary>
        public double Maxima { get; private set; } = double.NaN;

        /// <summary>
        /// Calculates a normalised output based in the given input.
        /// </summary>
        /// <param name="input">Input.</param>
        /// <returns>Normalised output.</returns>
        public double Calculate(double input)
        {
            if (double.IsNaN(Minima))
            {
                Minima = input;
            }
            if (double.IsNaN(Maxima))
            {
                Maxima = input;
            }

            if (input < Minima)
            {
                Minima = input;
            }
            else if (input > Maxima)
            {
                Maxima = input;
            }

            double result = 1.0;
            if (Minima != Maxima)
            {
                result = (input - Minima) / (Maxima - Minima);
            }
            return result;
        }
    }
}
