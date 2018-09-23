namespace RichTea.NeuralNetLib.Resizers
{
    /// <summary>
    /// Interface for changing the number of hidden layers a net has.
    /// </summary>
    interface ILayerResizer
    {
        /// <summary>
        /// Creates a new net from the given net with the new number of hidden layers.
        /// </summary>
        /// <param name="net">Source net.</param>
        /// <param name="hiddenLayerCount">Number of hidden layers layers the new net should have.</param>
        /// <returns>Net</returns>
        Net ResizeLayers(Net net, int hiddenLayerCount);
    }
}
