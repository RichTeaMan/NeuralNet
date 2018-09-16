using System;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// The result of a back propgation training cycle.
    /// </summary>
    /// <typeparam name="T">Type of net being trained.</typeparam>
    public class BackPropagationResult<T> where T : class
    {
        /// <summary>
        /// Gets the net that has been trained.
        /// </summary>
        public T Net { get; }

        /// <summary>
        /// Gets the SSE (sum of square of errors) of the net.
        /// </summary>
        public double SSE { get; }

        /// <summary>
        /// Initialises a back propagation result.
        /// </summary>
        /// <param name="net">Net that was trained.</param>
        /// <param name="sse">SSE.</param>
        public BackPropagationResult(T net, double sse)
        {
            Net = net ?? throw new ArgumentNullException(nameof(net));
            SSE = sse;
        }
    }
}
