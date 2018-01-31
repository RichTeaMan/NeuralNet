namespace RichTea.NeuralNetLib.Mutators
{
    public interface INeuralNetOneParentMutator : INeuralNetMutator
    {
        Net GenetateMutatedNeuralNet(Net parentNet);
    }
}
