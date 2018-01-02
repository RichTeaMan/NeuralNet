using Common;
using System;
using System.Linq;

namespace NeuralNetLib.Serialisation
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
                .Equals();
        }

        public override int GetHashCode()
        {
            return new HashCodeBuilder<SerialisedNodeLayer>(this)
                .Append(p => p.Nodes)
                .HashCode;
        }
    }
}
