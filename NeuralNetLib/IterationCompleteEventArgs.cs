using System;
using System.Collections.Generic;
using System.Text;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Iteration complete event arguments.
    /// </summary>
    public class IterationCompleteEventArgs : EventArgs
    {
        /// <summary>
        /// Gets a readonly list of evaluated nets.
        /// </summary>
        public IReadOnlyList<EvaluatedNet> EvaluatedNets { get; }

        /// <summary>
        /// Gets the training status.
        /// </summary>
        public TrainingStatus TrainingStatus { get; }

        /// <summary>
        /// Constructs iteration complete event arguments.
        /// </summary>
        /// <param name="evaluatedNets">Nets evalutated.</param>
        /// <param name="trainingStatus">Training status.</param>
        public IterationCompleteEventArgs(IReadOnlyList<EvaluatedNet> evaluatedNets, TrainingStatus trainingStatus)
        {
            EvaluatedNets = evaluatedNets;
            TrainingStatus = trainingStatus;
        }
    }
}
