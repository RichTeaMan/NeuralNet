using System;
using System.Collections.Generic;
using System.Text;

namespace RichTea.NeuralNetLib
{
    public class NormaliserNode
    {
        public double Minima { get; private set; } = 0;

        public double Maxima { get; private set; } = 0;

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
