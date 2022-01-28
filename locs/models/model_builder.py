import os

from locs.models.model_factory import NetworkFactory


def build_model(params):
    if params['model_type'] == 'nri':
        num_vars = params['num_vars']
        graph_type = params['graph_type']

        from locs.models import encoders
        from locs.models import decoders
        from locs.models import nri

        # Build Encoder
        encoder = encoders.RefMLPEncoder(params)
        print("ENCODER: ", encoder)

        # Build Decoder
        decoder = decoders.GraphRNNDecoder(params)
        print("DECODER: ", decoder)
        if graph_type == 'dynamic':
            model = nri.DynamicNRI(num_vars, encoder, decoder, params)
        else:
            model = nri.StaticNRI(num_vars, encoder, decoder, params)

    else:
        # dNRI, EGNN, GRU, and LoCS are implemented this way
        dynamic_vars = params.get('dynamic_vars', False)
        model = NetworkFactory.create(params['model_type'], dynamic_vars, params)
        print(f"{params['model_type']} MODEL: ", model)

    if params['load_best_model']:
        print("LOADING BEST MODEL")
        path = os.path.join(params['working_dir'], 'best_model')
        model.load(path)
    elif params['load_model']:
        print("LOADING MODEL FROM SPECIFIED PATH")
        model.load(params['load_model'])
    if params['gpu']:
        model.cuda()
    return model
