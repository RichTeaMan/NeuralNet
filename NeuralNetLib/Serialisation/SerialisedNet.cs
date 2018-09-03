using RichTea.Common;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib.Serialisation
{
    public class SerialisedNet
    {

        public SerialisedNodeLayer[] NodeLayers { get; set; }

        public Net CreateNet()
        {
            var nodeLayers = NodeLayers.Select(n => n.CreateNodeLayer());
            var net = new Net(nodeLayers);
            return net;
        }

        public override string ToString()
        {
            return new ToStringBuilder<SerialisedNet>(this)
                .Append(p => p.NodeLayers)
                .ToString();
        }

        public override bool Equals(object that)
        {
            return new EqualsBuilder<SerialisedNet>(this, that)
                .Append(p => p.NodeLayers)
                .AreEqual;
        }

        public override int GetHashCode()
        {
            var hashCodeBuilder = new HashCodeBuilder();

            var weights = NodeLayers.SelectMany(nl => nl.Nodes.SelectMany(n => n.Weights).ToArray()).ToArray();
            var biases = NodeLayers.SelectMany(nl => nl.Nodes.Select(n => n.Bias)).ToArray();

            hashCodeBuilder.Append(weights).Append(biases);
            return hashCodeBuilder.HashCode;
        }

        public static bool operator ==(SerialisedNet lhs, SerialisedNet rhs)
        {
            if (ReferenceEquals(lhs, null))
            {
                return ReferenceEquals(rhs, null);
            }

            return lhs.Equals(rhs);
        }

        public static bool operator !=(SerialisedNet lhs, SerialisedNet rhs)
        {
            return !(lhs == rhs);
        }
    }
}
