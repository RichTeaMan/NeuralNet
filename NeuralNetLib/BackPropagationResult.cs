using System;

namespace RichTea.NeuralNetLib
{
    public class BackPropagationResult<T> where T : class
    {
        public T Net{get;}
        public double SSE { get; }

        public BackPropagationResult(T net, double sse)
        {
            Net = net ?? throw new ArgumentNullException(nameof(net));
            SSE = sse;
        }
    }
}
