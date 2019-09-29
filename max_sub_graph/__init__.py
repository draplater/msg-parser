from . import cost_augment, decoder

graph_decoders = {"arcfactor": decoder.arcfactor,
            "1ec2p": decoder.oneec2p,
            "1ec2p-vine": decoder.oneec2p_vine}
