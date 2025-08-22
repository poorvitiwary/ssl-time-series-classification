"""Best parameters post optimisation for noisy learning."""

BEST_PARAMS = { 0.0318 : {'lr': 0.004063255837793846, 'nf': 211, 'bs': 64, 'fc_dropout': 0.13812932470456085, 'conv_dropout': 0.1276428059518422, 'ks': 57, 'bottleneck': False, 'coord': False, 'separable': False, 'dilation': 3, 'l2pl_ratio': 5},
0.0645: {'lr': 0.009923877405741348, 'nf': 228, 'bs': 128, 'fc_dropout': 0.40002244187730096, 'conv_dropout': 0.11014016712470376, 'ks': 59, 
               'bottleneck': False, 'coord': True, 'separable': True, 'dilation': 3, 'l2pl_ratio': 3
},

 0.1928: {'lr': 0.00219403586304545, 'nf': 121, 'bs': 64, 'fc_dropout': 0.24983971690679505, 'conv_dropout': 0.12232348367994114, 'ks': 53, 
               'bottleneck': False, 'coord': False, 'separable': True, 'dilation': 3, 'l2pl_ratio': 4
},

0.4039: {'lr': 0.0015840878111510863, 'nf': 185, 'bs': 48, 'fc_dropout': 0.4824722310317245, 'conv_dropout': 0.11429302905991293, 'ks': 36, 'bottleneck': False, 'coord': False, 'separable': True, 'dilation': 4, 'l2pl_ratio': 2},

0.5: {'lr': 0.0006869758846708765, 'nf': 147, 'bs': 16, 'fc_dropout': 0.4198478184741798, 'conv_dropout': 0.17195803735784618, 'ks': 56, 
               'bottleneck': False, 'coord': True, 'separable': True, 'dilation': 1, 'l2pl_ratio': 3
},

0.645: {'lr': 0.0027022534187512775, 'nf': 229, 'bs': 16, 'fc_dropout': 0.4208829736028663, 'conv_dropout': 0.13205986002376474, 'ks': 49, 
               'bottleneck': False, 'coord': False, 'separable': True, 'dilation': 3, 'l2pl_ratio': 2
}}