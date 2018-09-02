namespace RichTea.NeuralNetLib.Mutators
{
    /// <summary>
    /// Interface for mutator that one requires one parent neural net.
    /// </summary>
    public interface INeuralNetOneParentMutator : INeuralNetMutator
    {
        /// <summary>
        /// Create new nueral based upon a single parent.
        /// </summary>
        /// <param name="parentNet">Parent net.</param>
        /// <returns>Child net derived from parent net.</returns>
        Net GenetateMutatedNeuralNet(Net parentNet);
    }
}
