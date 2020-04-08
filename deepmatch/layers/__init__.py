from deepctr.layers import custom_objects
from deepctr.layers.utils import reduce_sum

from .core import PoolingLayer, SampledSoftmaxLayer, Similarity, LabelAwareAttention, CapsuleLayer,SampledSoftmaxLayerv2
from ..utils import sampledsoftmaxloss

_custom_objects = {'PoolingLayer': PoolingLayer,
                   'SampledSoftmaxLayer': SampledSoftmaxLayer,
                   'Similarity': Similarity,
                   'LabelAwareAttention': LabelAwareAttention,
                   'CapsuleLayer': CapsuleLayer,
                   'reduce_sum':reduce_sum,
                   'SampledSoftmaxLayerv2':SampledSoftmaxLayerv2,
                   'sampledsoftmaxloss':sampledsoftmaxloss
                   }

custom_objects = dict(custom_objects, **_custom_objects)
