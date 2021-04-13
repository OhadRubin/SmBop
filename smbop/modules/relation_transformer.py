from allennlp.modules.seq2seq_encoders.seq2seq_encoder import Seq2SeqEncoder
from torch import nn
import smbop.modules.transformer as transformer
from overrides import overrides


@Seq2SeqEncoder.register("relation_transformer")
class RelationTransformer(Seq2SeqEncoder):
    def __init__(
        self,
        num_layers=8,
        num_heads=8,
        hidden_size=768,
        ff_size=768,
        tie_layers=False,
        dropout=0.1,
        n_relations=51,
        tfixup=False,
        mu=31.8,
    ):
        super().__init__()
        if tfixup:
            use_layernorm = False
            self.mu = mu
        else:
            use_layernorm = True
        self.num_layers = num_layers
        self.encoder = transformer.Encoder(
            lambda: transformer.EncoderLayer(
                hidden_size,
                transformer.MultiHeadedAttentionWithRelations(
                    num_heads, hidden_size, dropout
                ),
                transformer.PositionwiseFeedForward(hidden_size, ff_size, dropout),
                n_relations,
                dropout,
                use_layernorm,
            ),
            hidden_size,
            num_layers,
            tie_layers,
            use_layernorm,
        )
        self._input_dim = hidden_size
        if tfixup:
            self.fixup_initialization()

    @overrides
    def get_input_dim(self) -> int:
        return self._input_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._input_dim

    @overrides
    def is_bidirectional(self):
        return False

    @overrides
    def forward(self, enc, relation, mask):
        return self.encoder(enc, relation, mask)

    def fixup_initialization(self):
        temp_state_dic = {}
        weights = [
            "w_1.weight",
            "w_2.weight",
            "self_attn.linears.2.weight",
            "self_attn.linears.3.weight",
            "relation_v_emb.weight",
        ]
        for name, param in self.named_parameters():
            if any(name.endswith(w) for w in weights):
                # print(name)
                temp_state_dic[name] = (
                    (self.num_layers * (4 * self.mu ** 2 + 2 * self.mu + 2))
                    ** (-1.0 / 2.0)
                ) * param
        for name in self.state_dict():
            if name not in temp_state_dic:
                temp_state_dic[name] = self.state_dict()[name]
        self.load_state_dict(temp_state_dic)
