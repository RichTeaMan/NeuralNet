using System;
using System.Collections.Generic;
using System.Text;

namespace RichTea.NeuralNetLib
{
    public class IterationInProgressEventArgs : EventArgs
    {
        public TrainingStatus TrainingStatus { get; }

        public IterationInProgressEventArgs(TrainingStatus trainingStatus)
        {
            TrainingStatus = trainingStatus;
        }
    }
}
