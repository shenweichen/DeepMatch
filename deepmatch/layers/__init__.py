from deepctr.layers import custom_objects
from deepctr.layers.utils import reduce_sum

from .core import PoolingLayer, LabelAwareAttention, CapsuleLayer, SampledSoftmaxLayer, EmbeddingIndex, \
    MaskUserEmbedding, InBatchSoftmaxLayer
from .interaction import DotAttention, ConcatAttention, SoftmaxWeightedSum, AttentionSequencePoolingLayer, \
    SelfAttention, \
    SelfMultiHeadAttention, UserAttention
from .sequence import DynamicMultiRNN
from ..utils import sampledsoftmaxloss

_custom_objects = {'PoolingLayer': PoolingLayer,
                   'LabelAwareAttention': LabelAwareAttention,
                   'CapsuleLayer': CapsuleLayer,
                   'reduce_sum': reduce_sum,
                   'SampledSoftmaxLayer': SampledSoftmaxLayer,
                   'InBatchSoftmaxLayer': InBatchSoftmaxLayer,
                   'sampledsoftmaxloss': sampledsoftmaxloss,
                   'EmbeddingIndex': EmbeddingIndex,
                   'DotAttention': DotAttention,
                   'ConcatAttention': ConcatAttention,
                   'SoftmaxWeightedSum': SoftmaxWeightedSum,
                   'AttentionSequencePoolingLayer': AttentionSequencePoolingLayer,
                   'SelfAttention': SelfAttention,
                   'SelfMultiHeadAttention': SelfMultiHeadAttention,
                   'UserAttention': UserAttention,
                   'DynamicMultiRNN': DynamicMultiRNN,
                   'MaskUserEmbedding': MaskUserEmbedding
                   }

custom_objects = dict(custom_objects, **_custom_objects)
