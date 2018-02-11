using System;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Iteration started event arguments.
    /// </summary>
    public class IterationStartedEventArgs : EventArgs
    {
        /// <summary>
        /// Gets training status.
        /// </summary>
        public TrainingStatus TrainingStatus { get; }

        /// <summary>
        /// Constructs iteration status event args.
        /// </summary>
        /// <param name="trainingStatus">Training status.</param>
        public IterationStartedEventArgs(TrainingStatus trainingStatus)
        {
            TrainingStatus = trainingStatus;
        }
    }
}
