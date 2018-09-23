using System;
using System.Linq;

namespace RichTea.NeuralNetLib.Resizers
{
    /// <summary>
    /// Creates nets with the specified number of layers. The weights for new nodes are seeded randomly.
    /// </summary>
    public class RandomLayerResizer : ILayerResizer
    {
        /// <summary>
        /// Random.
        /// </summary>
        private readonly Random _random;

        /// <summary>
        /// Initialises random layer resizer from a random.
        /// </summary>
        /// <param name="random">Random.</param>
        public RandomLayerResizer(Random random)
        {
            _random = random ?? throw new ArgumentNullException(nameof(random));
        }

        /// <summary>
        /// Initialises random layer resizer.
        /// </summary>
        public RandomLayerResizer() : this(new Random()) { }

        /// <summary>
        /// Resizes layers by creating new nodes.
        /// </summary>
        /// <param name="net">Source net.</param>
        /// <param name="hiddenLayerCount">Number of hidden layers the net should have.</param>
        /// <returns>Net</returns>
        public Net ResizeLayers(Net net, int hiddenLayerCount)
        {
            int layersToAdd = hiddenLayerCount - (net.Layers - 2);

            Net resultNet;

            if (layersToAdd >= 0)
            {
                int lastHiddenIndex = net.Layers - 2;
                var lastHidden = net.NodeLayers[lastHiddenIndex];
                var newLayers = Enumerable.Range(0, layersToAdd).Select(i => new NodeLayer(lastHidden.InputCount, lastHidden.OutputCount, _random));

                var layers = net.NodeLayers.Take(lastHiddenIndex + 1).Concat(newLayers).Concat(new[] { net.NodeLayers.Last() });
                resultNet = new Net(layers);
            }
            else
            {
                var layers = net.NodeLayers.Take(hiddenLayerCount + 1).Concat(new[] { net.NodeLayers.Last() });
                resultNet = new Net(layers);
            }

            return resultNet;
        }
    }
}
