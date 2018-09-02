namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Interface for mutator that one requires two parent neural nets.
    /// </summary>
    public interface INeuralNetTwoParentMutator : INeuralNetMutator
    {
        /// <summary>
        /// Creates child neural net from two parents.
        /// </summary>
        /// <param name="firstParentNet">First parent neural net.</param>
        /// <param name="secondParentNet">Second parent neural net.</param>
        /// <returns>Neurla net derived from the two parents.</returns>
        Net GenetateMutatedNeuralNet(Net firstParentNet, Net secondParentNet);
    }
}
