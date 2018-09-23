namespace RichTea.NeuralNetLib.Resizers
{
    /// <summary>
    /// Interface for changing the number of inputs a net has.
    /// </summary>
    interface IInputResizer
    {
        /// <summary>
        /// Creates a new net from the given net with the new number of inputs.
        /// </summary>
        /// <param name="net">Source net.</param>
        /// <param name="inputNumber">Number of inputs the new net should have.</param>
        /// <returns>Net</returns>
        Net ResizeInputs(Net net, int inputNumber);
    }
}
