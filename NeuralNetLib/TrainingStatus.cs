using System;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Represents the current status of the trainer.
    /// </summary>
    public class TrainingStatus
    {
        /// <summary>
        /// Gets the generation currently being processed.
        /// </summary>
        public int CurrentIteration { get; }

        /// <summary>
        /// Gets the number of evalutions that have so far occurred in this generation.
        /// </summary>
        public int GenerationEvaluations { get; }

        /// <summary>
        /// Gets the number of evalutations that have occurred in total.
        /// </summary>
        public int TotalEvaluations { get; }

        /// <summary>
        /// Gets the amount of time spent in the current generation.
        /// </summary>
        public TimeSpan GenerationTimeSpan { get; }

        /// <summary>
        /// Gets the total amount of time spent in total.
        /// </summary>
        public TimeSpan TotalTimeSpan { get; }

        /// <summary>
        /// Initialises training status.
        /// </summary>
        /// <param name="currentIteration">Current iteration.</param>
        /// <param name="generationEvaluations">Generation evaluations.</param>
        /// <param name="totalEvaluations">Total evaluations.</param>
        /// <param name="generationTimeSpan">Generation time span.</param>
        /// <param name="totalTimeSpan">Total time span.</param>
        public TrainingStatus(int currentIteration, int generationEvaluations, int totalEvaluations, TimeSpan generationTimeSpan, TimeSpan totalTimeSpan)
        {
            CurrentIteration = currentIteration;
            GenerationEvaluations = generationEvaluations;
            TotalEvaluations = totalEvaluations;
            GenerationTimeSpan = generationTimeSpan;
            TotalTimeSpan = totalTimeSpan;
        }
    }
}
