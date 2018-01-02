
using System;

namespace NeuralNetLib
{
    public static class RandomExtension
    {

        public static double NextDouble(this Random random, double min, double max)
        {
            double delta = max - min;
            var next = delta * random.NextDouble();
            var result = next + min;
            return result;
        }
        public static double NextDouble(this Random random, double range)
        {
            var result = random.NextDouble(-range, range);
            return result;
        }


        /// <summary>
        /// Gets a random double from the start value within +/- the supplied range.
        /// </summary>
        /// <param name="random"></param>
        /// <param name="start"></param>
        /// <param name="range"></param>
        /// <returns></returns>
        public static double NextDoubleInRange(this Random random, double start, double range)
        {
            var result = NextDouble(random, start - range, start + range);
            return result;
        }
    }
}
