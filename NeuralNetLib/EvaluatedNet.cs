using System;
using System.Collections.Generic;
using System.Text;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Stores results from <see cref="GeneticAlgorithmTrainer{T}"/>.
    /// </summary>
    public class EvaluatedNet
    {
        /// <summary>
        /// Gets the net that was evalutated.
        /// </summary>
        public Net Net { get; }

        /// <summary>
        /// Gets the score from the trainer's <see cref="IFitnessEvaluator"/>.
        /// </summary>
        public int FitnessScore { get; }

        /// <summary>
        /// Initialises evaluated net.
        /// </summary>
        /// <param name="net">The net that was evaluated.</param>
        /// <param name="fitnessScore">Fitness score of the net.</param>
        public EvaluatedNet(Net net, int fitnessScore)
        {
            Net = net;
            FitnessScore = fitnessScore;
        }
    }
}
