using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib
{
    public class Net : INet
    {
        #region Properties

        public INodeLayer[] NodeLayers { get; private set; }
        public int Inputs { get; private set; }
        public int Outputs { get; private set; }
        public int Layers
        {
            get { return NodeLayers.Length; }
        }

        #endregion

        #region Methods

        public Net(int Inputs, int Outputs, int Layers = 3)
        {
            if (Layers < 2)
                throw new ArgumentException("There must be at least 2 layers.");

            this.Inputs = Inputs;
            this.Outputs = Outputs;

            NodeLayers = new NodeLayer[Layers];
            for (int i = 0; i < Layers - 1; i++)
            {
                NodeLayers[i] = new NodeLayer(Inputs, Inputs);
            }
            NodeLayers[Layers - 1] = new NodeLayer(Inputs, Outputs);
        }

        public double[] Calculate(double[] Inputs)
        {
            if (this.Inputs != Inputs.Length)
                throw new ArgumentException("There is an incorrect number of Inputs.");

            double[] interStep = Inputs;
            foreach (var layer in NodeLayers)
            {
                interStep = layer.Calculate(interStep);
            }
            return interStep;
        }

        public double[] Calculate(double[] Inputs, double[] Targets, ref double SSE)
        {
            if(Targets.Length != Outputs)
                throw new ArgumentException("There is an incorrect number of Targets.");

            var results = Calculate(Inputs);

            SSE = 0;
            for (int i = 0; i < results.Length; i++)
            {
                double error = Targets[i] - results[i];
                SSE += Math.Pow(error, 2);
            }

            return results;
        }

        #endregion

    }
}
