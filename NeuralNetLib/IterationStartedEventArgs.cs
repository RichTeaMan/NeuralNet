using System;
using System.Collections.Generic;
using System.Text;

namespace RichTea.NeuralNetLib
{
    public class IterationStartedEventArgs : EventArgs
    {
        public TrainingStatus TrainingStatus { get; }

        public IterationStartedEventArgs(TrainingStatus trainingStatus)
        {
            TrainingStatus = trainingStatus;
        }
    }
}
