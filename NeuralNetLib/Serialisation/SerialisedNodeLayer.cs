using RichTea.Common;
using System.Collections.Generic;
using System.Linq;

namespace RichTea.NeuralNetLib.Serialisation
{
    public class SerialisedNodeLayer
    {
        public SerialisedNode[] Nodes { get; set; }

        public NodeLayer CreateNodeLayer()
        {
            var nodes = Nodes.Select(n => n.CreateNode()).ToArray();

            var nodeLayer = new NodeLayer(nodes);
            return nodeLayer;
        }

        public override string ToString()
        {
            return new ToStringBuilder<SerialisedNodeLayer>(this)
                .Append(p => p.Nodes)
                .ToString();
        }

        public override bool Equals(object that)
        {
            return new EqualsBuilder<SerialisedNodeLayer>(this, that)
                .Append(p => p.Nodes)
                .AreEqual;
        }

        public override int GetHashCode()
        {
            var hashCodeBuilder = new HashCodeBuilder()
                   .Append(Nodes);

            return hashCodeBuilder.HashCode;
        }

        public static bool operator ==(SerialisedNodeLayer lhs, SerialisedNodeLayer rhs)
        {
            if (ReferenceEquals(lhs, null))
            {
                return ReferenceEquals(rhs, null);
            }

            return lhs.Equals(rhs);
        }

        public static bool operator !=(SerialisedNodeLayer lhs, SerialisedNodeLayer rhs)
        {
            return !(lhs == rhs);
        }
    }
}
