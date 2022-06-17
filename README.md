# DeepMatch

[![Python Versions](https://img.shields.io/pypi/pyversions/deepmatch.svg)](https://pypi.org/project/deepmatch)
[![TensorFlow Versions](https://img.shields.io/badge/TensorFlow-1.4+/2.0+-blue.svg)](https://pypi.org/project/deepmatch)
[![PyPI Version](https://img.shields.io/pypi/v/deepmatch.svg)](https://pypi.org/project/deepmatch)
[![GitHub Issues](https://img.shields.io/github/issues/shenweichen/deepmatch.svg
)](https://github.com/shenweichen/deepmatch/issues)
<!-- [![Activity](https://img.shields.io/github/last-commit/shenweichen/deepmatch.svg)](https://github.com/shenweichen/DeepMatch/commits/master) -->


[![Documentation Status](https://readthedocs.org/projects/deepmatch/badge/?version=latest)](https://deepmatch.readthedocs.io/)
![CI status](https://github.com/shenweichen/deepmatch/workflows/CI/badge.svg)
[![codecov](https://codecov.io/gh/shenweichen/DeepMatch/branch/master/graph/badge.svg)](https://codecov.io/gh/shenweichen/DeepMatch)
[![Disscussion](https://img.shields.io/badge/chat-wechat-brightgreen?style=flat)](./README.md#disscussiongroup)
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

## DisscussionGroup & Related Projects

<html>
    <table style="margin-left: 20px; margin-right: auto;">
        <tr>
            <td>
                公众号：<b>浅梦的学习笔记</b><br><br>
                <a href="https://github.com/shenweichen/deepmatch">
  <img align="center" src="./docs/pics/code.png" />
</a>
            </td>
            <td>
                微信：<b>deepctrbot</b><br><br>
 <a href="https://github.com/shenweichen/deepmatch">
  <img align="center" src="./docs/pics/deepctrbot.png" />
</a>
            </td>
            <td>
<ul>
<li><a href="https://github.com/shenweichen/AlgoNotes">AlgoNotes</a></li>
<li><a href="https://github.com/shenweichen/DeepCTR">DeepCTR</a></li>
<li><a href="https://github.com/shenweichen/DeepCTR-Torch">DeepCTR-Torch</a></li>
<li><a href="https://github.com/shenweichen/GraphEmbedding">GraphEmbedding</a></li>
</ul>
            </td>
        </tr>
    </table>
</html>
