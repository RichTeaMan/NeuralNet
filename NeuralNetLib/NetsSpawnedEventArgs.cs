using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace RichTea.NeuralNetLib
{
    public class NetsSpawnedEventArgs : EventArgs
    {
        public IReadOnlyList<Net> Nets { get; }

        public NetsSpawnedEventArgs(IEnumerable<Net> nets)
        {
            Nets = nets.ToList();
        }
    }
}
