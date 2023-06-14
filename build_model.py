from networks import TransformerEncoder, FedAlign, ConventionalClassifier


def build_model(use_label_encoder, hidden_dim, data_feature_size, n_class, nhead, num_encoder_layers, dim_feedforward, dropout, pretrained_embedding=None, do_input_embedding=False):
    data_encoder = TransformerEncoder(
        d_model=hidden_dim,
        in_vocab_size=data_feature_size,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout,
        do_input_embedding=do_input_embedding
    )

    if use_label_encoder:
        model = FedAlign(data_encoder, emb_dim=hidden_dim, out_dim=n_class, pretrained_embedding=pretrained_embedding)
    else:
        model = ConventionalClassifier(data_encoder, emb_dim=hidden_dim, out_dim=n_class)

    return model