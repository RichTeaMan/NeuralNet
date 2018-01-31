namespace RichTea.NeuralNetLib.Mutators
{
    public interface INeuralNetTwoParentMutator : INeuralNetMutator
    {
        Net GenetateMutatedNeuralNet(Net firstParentNet, Net secondParentNet);
    }
}
