using Common;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace NeuralNetLib.Serialisation
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
                .Equals();
        }

        public override int GetHashCode()
        {
            return new HashCodeBuilder<SerialisedNet>(this)
                .Append(p => p.NodeLayers)
                .HashCode;
        }
    }
}
