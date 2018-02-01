using System;
using System.Collections.Generic;
using System.Text;

namespace RichTea.NeuralNetLib
{
    public class IterationCompleteEventArgs : EventArgs
    {
        public IReadOnlyList<EvaluatedNet> EvaluatedNets { get; }

        public TrainingStatus TrainingStatus { get; }

        public IterationCompleteEventArgs(IReadOnlyList<EvaluatedNet> evaluatedNets, TrainingStatus trainingStatus)
        {
            EvaluatedNets = evaluatedNets;
            TrainingStatus = trainingStatus;
        }
    }
}
