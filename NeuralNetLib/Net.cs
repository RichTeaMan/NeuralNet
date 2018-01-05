using RichTea.NeuralNetLib.Serialisation;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RichTea.NeuralNetLib
{
    public class Net
    {
        #region Properties

        public NodeLayer[] NodeLayers { get; private set; }
        public int Inputs { get; private set; }
        public int Outputs { get; private set; }
        public int Layers
        {
            get { return NodeLayers.Length; }
        }

        #endregion

        #region Methods

        public Net(int inputs, int outputs, int layers = 3)
        {
            if (layers < 2)
            {
                throw new ArgumentException("There must be at least 2 layers.");
            }

            Inputs = inputs;
            Outputs = outputs;

            NodeLayers = new NodeLayer[layers];
            for (int i = 0; i < layers - 1; i++)
            {
                NodeLayers[i] = new NodeLayer(inputs, inputs);
            }
            NodeLayers[layers - 1] = new NodeLayer(inputs, outputs);
        }

        public Net(IEnumerable<NodeLayer> nodeLayers)
        {
            if (nodeLayers.Count() < 2)
            {
                throw new ArgumentException("There must be at least 2 layers.");
            }

            Inputs = nodeLayers.First().Inputs;
            Outputs = nodeLayers.Last().Outputs;

            NodeLayers = nodeLayers.ToArray();
        }

        public void SeedWeights(Random random)
        {
            foreach (var nodeLayer in NodeLayers)
            {
                nodeLayer.SeedWeights(random);
            }
        }

        public SerialisedNet CreateSerialisedNet()
        {
            var serialisedNodeLayers = NodeLayers.Select(n => n.CreateSerialisedNodeLayer()).ToArray();
            var serialisedNet = new SerialisedNet
            {
                NodeLayers = serialisedNodeLayers
            };
            return serialisedNet;
        }

        public void SeedWeights(Net net)
        {
            if (net.Inputs != Inputs)
            {
                throw new ArgumentException("Net has incorrect number of inputs.");
            }
            if (net.Outputs != Outputs)
            {
                throw new ArgumentException("Net has incorrect number of outputs.");
            }

            for (int i = 0; i < NodeLayers.Length; i++)
            {
                NodeLayers[i].SeedWeights(net.NodeLayers[i]);
            }
        }

        public double[] Calculate(double[] inputs)
        {
            if (Inputs != inputs.Length)
            {
                throw new ArgumentException("There is an incorrect number of Inputs.");
            }

            double[] interStep = inputs;
            foreach (var layer in NodeLayers)
            {
                interStep = layer.Calculate(interStep);
            }
            return interStep;
        }

        public double[] Calculate(double[] inputs, double[] targets, ref double sse)
        {
            if (targets.Length != Outputs)
            {
                throw new ArgumentException("There is an incorrect number of Targets.");
            }

            var results = Calculate(inputs);

            sse = 0;
            for (int i = 0; i < results.Length; i++)
            {
                double error = targets[i] - results[i];
                sse += Math.Pow(error, 2);
            }

            return results;
        }

        #endregion

    }
}
