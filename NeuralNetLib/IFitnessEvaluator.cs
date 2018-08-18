using System;
using System.Collections.Generic;

namespace RichTea.NeuralNetLib
{
    /// <summary>
    /// Interface for net fitness evaluation.
    /// </summary>
    public interface IFitnessEvaluator
    {
        /// <summary>
        /// Interface for extracting fitness score from a net. An implemting class will
        /// have this method called at least once per net for every generation.
        /// 
        /// Int.MaxValue is the considered the most fit, Int.MinValue is the last fit.
        /// Ideally, evaluations should be as deterministic as possible and return the same
        /// result everytime for a given net.
        /// </summary>
        /// <param name="competingNets">Competing nets.</param>
        /// <param name="evaluatingNet">Evaluating net.</param>
        /// <param name="trainingStatus">Training status.</param>
        /// <returns></returns>
        int EvaluateNet(IReadOnlyList<Net> competingNets, Net evaluatingNet, TrainingStatus trainingStatus);
    }
}
