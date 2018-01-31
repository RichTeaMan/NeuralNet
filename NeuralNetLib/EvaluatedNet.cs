using System;
using System.Collections.Generic;
using System.Text;

namespace RichTea.NeuralNetLib
{
    public class EvaluatedNet
    {
        public Net Net { get; }

        public int FitnessScore { get; }

        public EvaluatedNet(Net net, int fitnessScore)
        {
            Net = net;
            FitnessScore = fitnessScore;
        }
    }
}
