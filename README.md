# DeepMatch

[![Python Versions](https://img.shields.io/pypi/pyversions/deepmatch.svg)](https://pypi.org/project/deepmatch)
[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-1.9+/2.0+-blue.svg)](https://pypi.org/project/deepmatch)
[![Downloads](https://pepy.tech/badge/deepmatch)](https://pepy.tech/project/deepmatch)
[![PyPI Version](https://img.shields.io/pypi/v/deepmatch.svg)](https://pypi.org/project/deepmatch)
[![GitHub Issues](https://img.shields.io/github/issues/shenweichen/deepmatch.svg
)](https://github.com/shenweichen/deepmatch/issues)
<!-- [![Activity](https://img.shields.io/github/last-commit/shenweichen/deepmatch.svg)](https://github.com/shenweichen/DeepMatch/commits/master) -->


[![Documentation Status](https://readthedocs.org/projects/deepmatch/badge/?version=latest)](https://deepmatch.readthedocs.io/)
![CI status](https://github.com/shenweichen/deepmatch/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/shenweichen/DeepMatch/branch/master/graph/badge.svg)](https://codecov.io/gh/shenweichen/DeepMatch)
[![Codacy Badge](https://app.codacy.com/project/badge/Grade/c5a2769ec35444d8958f6b58ff85029b)](https://www.codacy.com/gh/shenweichen/DeepMatch/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=shenweichen/DeepMatch&amp;utm_campaign=Badge_Grade)
[![Disscussion](https://img.shields.io/badge/chat-wechat-brightgreen?style=flat)](https://github.com/shenweichen/DeepMatch#disscussiongroup)
[![License](https://img.shields.io/github/license/shenweichen/deepmatch.svg)](https://github.com/shenweichen/deepmatch/blob/master/LICENSE)

DeepMatch is a deep matching model library for recommendations & advertising. It's easy to **train models** and to **export representation vectors** for user and item which can be used for **ANN search**.You can use any complex model with `model.fit()`and `model.predict()` .

Let's [**Get Started!**](https://deepmatch.readthedocs.io/en/latest/Quick-Start.html) or [**Run examples**](./examples/colab_MovieLen1M_YoutubeDNN.ipynb) !



## Models List

|                 Model                  | Paper                                                                                                                                                           |
| :------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------- |
|  FM  | [ICDM 2010][Factorization Machines](https://www.researchgate.net/publication/220766482_Factorization_Machines) |
| DSSM | [CIKM 2013][Deep Structured Semantic Models for Web Search using Clickthrough Data](https://www.microsoft.com/en-us/research/publication/learning-deep-structured-semantic-models-for-web-search-using-clickthrough-data/)    |
| YoutubeDNN     | [RecSys 2016][Deep Neural Networks for YouTube Recommendations](https://www.researchgate.net/publication/307573656_Deep_Neural_Networks_for_YouTube_Recommendations)            |
| NCF  | [WWW 2017][Neural Collaborative Filtering](https://arxiv.org/abs/1708.05031)       |
| SDM  | [CIKM 2019][SDM: Sequential Deep Matching Model for Online Large-scale Recommender System](https://arxiv.org/abs/1909.00385)  |
| MIND | [CIKM 2019][Multi-interest network with dynamic routing for recommendation at Tmall](https://arxiv.org/pdf/1904.08030)  |
| COMIREC | [KDD 2020][Controllable Multi-Interest Framework for Recommendation](https://arxiv.org/pdf/2005.09347.pdf)  |

## Contributors([welcome to join us!](./CONTRIBUTING.md))

<table border="0">
  <tbody>
    <tr align="center" >
      <td>
        ​ <a href="https://github.com/shenweichen"><img width="70" height="70" src="https://github.com/shenweichen.png?s=40" alt="pic"></a><br>
        ​ <a href="https://github.com/shenweichen">Shen Weichen</a> ​
        <p>
        Alibaba Group  </p>​
      </td>
      <td>
         <a href="https://github.com/wangzhegeek"><img width="70" height="70" src="https://github.com/wangzhegeek.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/wangzhegeek">Wang Zhe</a> ​
        <p>Baidu Inc.  </p>​
      </td>
      <td>
        ​ <a href="https://github.com/clhchtcjj"><img width="70" height="70" src="https://github.com/clhchtcjj.png?s=40" alt="pic"></a><br>
        ​ <a href="https://github.com/clhchtcjj">Chen Leihui</a> ​
        <p>
        Alibaba Group  </p>​
      </td>
      <td>
        ​ <a href="https://github.com/LeoCai"><img width="70" height="70" src="https://github.com/LeoCai.png?s=40" alt="pic"></a><br>
         <a href="https://github.com/LeoCai">LeoCai</a>
         <p>  ByteDance   </p>​
      </td>
      <td>
        ​ <a href="https://github.com/liyuan97"><img width="70" height="70" src="https://github.com/liyuan97.png?s=40" alt="pic"></a><br>
        ​ <a href="https://github.com/liyuan97">Li Yuan</a>
        <p> Tencent    </p>​
      </td>
      <td>
        ​ <a href="https://github.com/yangjieyu"><img width="70" height="70" src="https://github.com/yangjieyu.png?s=40" alt="pic"></a><br>
        ​ <a href="https://github.com/yangjieyu">Yang Jieyu</a>
        <p> Ant Group    </p>​
      </td>
      <td>
        ​ <a href="https://github.com/zzszmyf"><img width="70" height="70" src="https://github.com/zzszmyf.png?s=40" alt="pic"></a><br>
        ​ <a href="https://github.com/zzszmyf">Meng Yifan</a>
        <p> DeepCTR    </p>​
      </td>
    </tr>
  </tbody>
</table>

## DisscussionGroup

- [Github Discussions](https://github.com/shenweichen/DeepMatch/discussions)
- Wechat Discussions

|公众号：浅梦学习笔记|微信：deepctrbot|学习小组 [加入](https://t.zsxq.com/026UJEuzv) [主题集合](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MjM5MzY4NzE3MA==&action=getalbum&album_id=1361647041096843265&scene=126#wechat_redirect)|
|:--:|:--:|:--:|
| [![公众号](./docs/pics/code.png)](https://github.com/shenweichen/AlgoNotes)| [![微信](./docs/pics/deepctrbot.png)](https://github.com/shenweichen/AlgoNotes)|[![学习小组](./docs/pics/planet_github.png)](https://t.zsxq.com/026UJEuzv)|

