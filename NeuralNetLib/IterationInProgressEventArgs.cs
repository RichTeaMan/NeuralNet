using System;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Iteration in progress event arguments.
    /// </summary>
    public class IterationInProgressEventArgs : EventArgs
    {
        /// <summary>
        /// Gets training status.
        /// </summary>
        public TrainingStatus TrainingStatus { get; }

        /// <summary>
        /// Constructs interation in progress event args.
        /// </summary>
        /// <param name="trainingStatus">Training status.</param>
        public IterationInProgressEventArgs(TrainingStatus trainingStatus)
        {
            TrainingStatus = trainingStatus;
        }
    }
}
