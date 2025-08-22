"""Best parameters post optimisation for fixmatch learning."""

BEST_PARAMS = { 0.0318 : {'lr': 0.00020771156145351792, 'nf': 157, 'bs': 128, 'fc_dropout': 0.47478717426699346, 'conv_dropout': 0.12520247249221358, 'ks': 54, 'bottleneck': True, 'coord': False, 'separable': False, 'dilation': 3},
0.0645: {'lr': 0.0006216007139347216, 'nf': 255, 'bs': 64, 'fc_dropout': 0.30929763733137083, 'conv_dropout': 0.185164021873564, 'ks': 58, 'bottleneck': False, 'coord': False, 'separable': False, 'dilation': 4},
0.129:{'lr': 0.005231693378181246, 'nf': 236, 'bs': 48, 'fc_dropout': 0.4243344532526303, 'conv_dropout': 0.18303827188759986, 'ks': 53, 'bottleneck': False, 'coord': False, 'separable': True, 'dilation': 4},
0.1928: {'lr': 0.0005642425561015764, 'nf': 253, 'bs': 48, 'fc_dropout': 0.3155432462780189, 'conv_dropout': 0.11811564340194823, 'ks': 34, 'bottleneck': False, 'coord': False, 'separable': False, 'dilation': 4},

0.4039: {'lr': 0.00016468040015173337, 'nf': 179, 'bs': 64, 'fc_dropout': 0.23417012682419402, 'conv_dropout': 0.13509113290273142, 'ks': 48, 'bottleneck': False, 'coord': False, 'separable': False, 'dilation': 4},

0.5: {'lr': 0.0010442730215767281, 'nf': 163, 'bs': 128, 'fc_dropout': 0.18905239922155248, 'conv_dropout': 0.12643726385509527, 'ks': 38, 'bottleneck': False, 'coord': False, 'separable': True, 'dilation': 3},

0.645:  {'lr': 0.0005016644027230172, 'nf': 245, 'bs': 128, 'fc_dropout': 0.12890595736155222, 'conv_dropout': 0.10627349478037414, 'ks': 25, 'bottleneck': True, 'coord': False, 'separable': True, 'dilation': 4}}