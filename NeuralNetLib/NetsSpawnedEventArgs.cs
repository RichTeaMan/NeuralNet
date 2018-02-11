using RichTea.NeuralNetLib.Mutators;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Nets spawned event arguments.
    /// </summary>
    public class NetsSpawnedEventArgs : EventArgs
    {
        /// <summary>
        /// Gets a readonly list of the nets that have just been created.
        /// </summary>
        public IReadOnlyList<Net> Nets { get; }

        /// <summary>
        /// Gets the mutator that spawned these nets. Note that if the nets have been spawned for the initial iteration this will be null.
        /// </summary>
        public INeuralNetMutator NeuralNetMutator { get; }

        /// <summary>
        /// Constructs net spawned event args.
        /// </summary>
        /// <param name="nets">List of nets.</param>
        /// <param name="neuralNetMutator">The mutator used.</param>
        public NetsSpawnedEventArgs(IEnumerable<Net> nets, INeuralNetMutator neuralNetMutator)
        {
            Nets = nets.ToList();
            NeuralNetMutator = neuralNetMutator;
        }
    }
}
