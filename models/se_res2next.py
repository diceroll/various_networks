import chainer
import chainer.functions as F
import chainer.links as L
from chainer import initializers
from chainercv.links import PickableSequentialChain

from models.connections.conv_2d_bn_activ import Conv2DBNActiv
from models.connections.resblock import ResBlock


class SERes2NeXt(PickableSequentialChain):

    _blocks = {
        50: [3, 4, 6, 3],
        101: [3, 4, 23, 3],
        152: [3, 8, 36, 3]
    }

    def __init__(self, n_layer, n_class=None,
                 initialW=None, fc_kwargs={},
                 scale=4, groups=4):
        blocks = self._blocks[n_layer]

        if initialW is None:
            initialW = initializers.HeNormal(scale=1., fan_option='fan_out')
        if 'initialW' not in fc_kwargs:
            fc_kwargs['initialW'] = initializers.Normal(scale=0.01)

        kwargs = {
            'scale': scale, 'groups': groups, 'initialW': initialW,
            'stride_first': False, 'add_block': 'se'}

        super(SERes2NeXt, self).__init__()
        with self.init_scope():
            self.conv1 = Conv2DBNActiv(None, 64, 7, 2, 3, nobias=True,
                                       initialW=initialW)
            self.pool1 = lambda x: F.max_pooling_2d(x, ksize=3, stride=2)
            self.res2 = ResBlock(blocks[0], None, 128, 256, 1, **kwargs)
            self.res3 = ResBlock(blocks[1], None, 256, 512, 2, **kwargs)
            self.res4 = ResBlock(blocks[2], None, 512, 1024, 2, **kwargs)
            self.res5 = ResBlock(blocks[3], None, 1024, 2048, 2, **kwargs)
            self.pool5 = lambda x: F.average(x, axis=(2, 3))
            self.fc6 = L.Linear(None, n_class, **fc_kwargs)


class SERes2NeXt50(SERes2NeXt):

    def __init__(self, n_class=None, initialW=None, fc_kwargs={}):
        super(SERes2NeXt50, self).__init__(
            50, n_class, initialW, fc_kwargs)


class SERes2NeXt101(SERes2NeXt):

    def __init__(self, n_class=None, initialW=None, fc_kwargs={}):
        super(SERes2NeXt101, self).__init__(
            101, n_class, initialW, fc_kwargs)


class SERes2NeXt152(SERes2NeXt):

    def __init__(self, n_class=None, initialW=None, fc_kwargs={}):
        super(SERes2NeXt152, self).__init__(
            152, n_class, initialW, fc_kwargs)
