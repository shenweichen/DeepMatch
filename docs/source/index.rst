.. DeepMatch documentation master file, created by
   sphinx-quickstart on Sun Apr  5 20:44:18 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to DeepMatch's documentation!
=====================================



DeepMatch is a  deep matching model library for recommendations, advertising, and search. It's easy to **train models** and to **export representation vectors** for user and item which can be used for **ANN search**.You can use any complex model with ``model.fit()`` and ``model.predict()`` .


Let's `Get Started! <./Quick-Start.html>`_  or `Run examples! <https://github.com/shenweichen/DeepMatch/blob/master/examples/colab_MovieLen1M_YoutubeDNN.ipynb>`_

You can read the latest code at https://github.com/shenweichen/DeepMatch

News
-----

10/12/2020 : Support different initializers for different embedding weights and loading pretrained embeddings. `Changelog <https://github.com/shenweichen/DeepMatch/releases/tag/v0.2.0>`_

05/17/2020 : Add ``SDM`` model. `Changelog <https://github.com/shenweichen/DeepMatch/releases/tag/v0.1.3>`_

04/10/2020 : Support `saving and loading model <./FAQ.html#save-or-load-weights-models>`_  . `Changelog <https://github.com/shenweichen/DeepMatch/releases/tag/v0.1.2>`_

DisscussionGroup
-----------------------

公众号：**浅梦的学习笔记**  wechat ID: **deepctrbot**

.. image:: ../pics/code.png

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
