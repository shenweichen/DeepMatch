.. DeepMatch documentation master file, created by
   sphinx-quickstart on Sun Apr  5 20:44:18 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepMatch's documentation!
=====================================

|Downloads|_ |Stars|_ |Forks|_ |PyPii|_ |Issues|_ |Chat|_

.. |Downloads| image:: https://pepy.tech/badge/deepmatch
.. _Downloads: https://pepy.tech/project/deepmatch

.. |Stars| image:: https://img.shields.io/github/stars/shenweichen/deepmatch.svg
.. _Stars: https://github.com/shenweichen/DeepMatch

.. |Forks| image:: https://img.shields.io/github/forks/shenweichen/deepmatch.svg
.. _Forks: https://github.com/shenweichen/DeepMatch/fork

.. |PyPii| image:: https://img.shields.io/pypi/v/deepmatch.svg
.. _PyPii: https://pypi.org/project/deepmatch

.. |Issues| image:: https://img.shields.io/github/issues/shenweichen/deepmatch.svg
.. _Issues: https://github.com/shenweichen/deepmatch/issues

.. |Chat| image:: https://img.shields.io/badge/chat-wechat-brightgreen?style=flat
.. _Chat: ./#disscussiongroup


DeepMatch is a  deep matching model library for recommendations, advertising, and search. It's easy to **train models** and to **export representation vectors** for user and item which can be used for **ANN search**.You can use any complex model with ``model.fit()`` and ``model.predict()`` .


Let's `Get Started! <./Quick-Start.html>`_  or `Run examples! <https://github.com/shenweichen/DeepMatch/blob/master/examples/colab_MovieLen1M_YoutubeDNN.ipynb>`_

You can read the latest code at https://github.com/shenweichen/DeepMatch

News
-----

10/31/2022 : Add `ComiRec` . `Changelog <https://github.com/shenweichen/DeepMatch/releases/tag/v0.3.1>`_

07/04/2022 : Support different negative sampling strategies, including `inbatch` , `uniform` , `frequency` , `adaptive` . `Changelog <https://github.com/shenweichen/DeepMatch/releases/tag/v0.3.0>`_

06/17/2022 : Fix some bugs. `Changelog <https://github.com/shenweichen/DeepMatch/releases/tag/v0.2.1>`_

DisscussionGroup
-----------------------


  公众号：**浅梦学习笔记**  wechat ID: **deepctrbot**

  `Discussions <https://github.com/shenweichen/DeepMatch/discussions>`_ `学习小组主题集合 <https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MjM5MzY4NzE3MA==&action=getalbum&album_id=1361647041096843265&scene=126#wechat_redirect>`_

.. image:: ../pics/code2.jpg


.. toctree::
   :maxdepth: 2
   :caption: Home:

   Quick-Start<Quick-Start.md>
   Features<Features.md>
   Examples<Examples.md>
   FAQ<FAQ.md>
   History<History.md>

.. toctree::
   :maxdepth: 3
   :caption: API:

   Models<Models>


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
