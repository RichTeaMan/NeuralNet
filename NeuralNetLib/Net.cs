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

        public Net(int Inputs, int Output, int Layers = 3)
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
            NodeLayers[Layers - 1] = new NodeLayer(Inputs, Output);
        }

        public double[] Calculate(double[] Inputs)
        {
            double[] interStep = Inputs;
            foreach (var layer in NodeLayers)
            {
                interStep = layer.Calculate(interStep);
            }
            return interStep;
        }

        public double[] Calculate(double[] Inputs, double[] Targets, ref double SSE)
        {
            throw new NotImplementedException();
        }

        #endregion

    }
}
