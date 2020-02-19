# [PyTorch로 딥러닝 시작하기](http://acornpub.co.kr/book/deep-learning-pytorch)

여기는 [PyTorch로 딥러닝 시작하기](http://acornpub.co.kr/book/deep-learning-pytorch)의 코드 레파지토리입니다. 원서인 [Deep Learning with PyTorch](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-pytorch?utm_source=github&utm_medium=repository&utm_campaign=9781788624336)는 [Packt](https://www.packtpub.com/?utm_source=github)에서 출간되었고, [에이콘](http://acornpub.co.kr)에서 번역되었습니다. 이 repository [PyTorch로 딥러닝 시작하기](http://acornpub.co.kr/book/deep-learning-pytorch) 실습에 필요한 모든 파일을 포함합니다.

[PyTorch로 딥러닝 시작하기](http://acornpub.co.kr/book/deep-learning-pytorch)의 보강 문서인 [](https://github.com/gkqlsdlek123/pytorch)와 이 레파지토리를 함께 참고하시기 바랍니다.

## [PyTorch로 딥러닝 시작하기](http://acornpub.co.kr/book/deep-learning-pytorch)에 대하여...

딥러닝를 통해서 구글 보이스 (Google Voice), 시리 (Siri), 알렉사 (Alexa)와 같은 세계에서 가장 지능적인 시스템은 더욱 강력해지고 있습니다. GPU, PyTorch, Keras, Tensorflow, CNTK 등의 소프트웨어 프레임워크와 빅데이터 그리고 강력한 하드웨어의 발전하면서 Text, Vision 및 고급 분석 분야의 여러 문제의 솔루션을보다 쉽게 구현할 수있게되었습니다.

이 책은 가장 진보적인 딥러닝 라이브러리인 PyToch로 소개합니다. PyTorch는 Python으로 작성되었으며, 접근성과 효율성이 뛰어나기 때문에, 데이터 과학 전문가들의 관심을 끌고 있습니다. PyTorch를 설치하는 것으로 시작하여 신속하게 다양한 통계 작업할 수 있습습니다. 앞으로 PyTorch로 신경망을 배우고 CNN, RNN 및 LSTM을 살펴 보겠습니다.

이 책은 ResNet, DenseNet, Inception 및 Seq2Seq와 같은 다양한 최신 딥러닝 아키텍처에 대한 직관을 제공합니다. 지나치게 수학에 의존하지 않습니다. 다음 책을 읽는 동안 GPU 컴퓨팅에 대해 배우게됩니다. PyTorch를 사용하여 모델을 교육하고 생성 네트워크와 같은 복잡한 신경 네트워크로 들어가 텍스트 및 이미지를 생성하는 방법을 살펴볼 것입니다.

이 책을 마치면 PyTorch로 딥러닝 애플리케이션에 쉽게 구현할 수 있습니다. 또한 PyTorch를 시작하고 실행하는 데 필요한 모든 정보를 얻을 수 있습니다.

## 예제 코드 구성

각 장의 코드는 폴더로 구분되어 있습니다. 각 폴더는 번호와 응용 프로그램 이름으로 시작됩니다. 예 : study02.

코드는 다음과 같은 형태로 제공됩니다.

```
x,y = get_data() # x - 학습 데이터,y - 목적변수
w,b = get_weights() # w,b - 학습 파라미터
for i in range(500):
y_pred = simple_network(x) # wx + b 연산 함수
loss = loss_fn(y,y_pred) # y와 y_pred의 차이 제곱 합
if i % 50 == 0:
print(loss)
optimize(learning_rate) # 오차를 최소화하도록 w, b를 조정
```

1장(딥러닝 시작하기) 9장(마지막 그리소 새로운 시작)을 제외한 2 ~8장은 모두 Jupyter 노트북이 제공됩니다. 책에서는 지면을 절약하기 위해서 코드를 실행하는 데 필요한 import 포함하지 않을 수 있습니다. 노트북에서는 모든 코드를 실행할 수 있습니다. 이 책은 실용적인 설명에 중점을두고 있으므로 책을 읽으면서 주피터 노트를 실행해 보시기 바랍니다. GPU가있는 컴퓨터에 액세스하면 코드를 빨리 실행할 수 있습니다. paperspace.com 및 www.crestle.com과 같은 회사는 딥러닝 알고리즘을 실행하는 많은 복잡성을 추상화하는 서비스를 제공합니다.   

## 관련 서적
* [Hands-On Deep Learning with PyTorch](https://www.packtpub.com/big-data-and-business-intelligence/hands-deep-learning-pytorch?utm_source=github&utm_medium=repository&utm_campaign=9781788834131)

* [Advanced Deep Learning with Keras](https://www.packtpub.com/big-data-and-business-intelligence/advanced-deep-learning-keras?utm_source=github&utm_medium=repository&utm_campaign=9781788629416)

* [Deep Learning with TensorFlow - Second Edition](https://www.packtpub.com/big-data-and-business-intelligence/deep-learning-tensorflow-second-edition?utm_source=github&utm_medium=repository&utm_campaign=9781788831109)

## 제안과 피드백
duddhks9244@gmail.com



# 2. [환경 ]

## 2.1.3 macOS
macOSX에 PyTorch 환경을 구성하는 과정을 소개합니다. 본 문서는 다음과 같은 형식으로 구성됩니다.

Python 3.7 설치
Anaconda 설치
Pytorch 실습을 위한 라이브러리 설치
1. Python 3.7 설치
macOS Mojava를 기준으로 Python3.7를 설치하는 방법을 소개합니다.

다음 URL에서 macOS용 Python 설치 프로그램을 다운로드 받습니다.

https://www.python.org/downloads/release/python-372/[↗NW]
위 URL에서 Mac용 설치 프로그램을 다운로드하면 다음과 같이 약 30MB의 PKG 파일이 다운로드됩니다. 기본값으로 설치를 진행합니다.

다음 명령으로 Python 3.7.2의 설치 상태를 확인합니다.

```
$ python3.7 -V
Python 3.7.2
$ python3 -V
Python 3.7.2
```



## 2. Anaconda 설치
다음 URL에서 Mac용 Anaconda 설치 프로그램을 다운로드 받습니다.
https://www.anaconda.com/distribution/#download-section
Mac용 Anaconda 설치 프로그램 크기는 684MB입니다.

Anaconda3를 설치하면 다음과 같이 환경 변수를 등록합니다.
```
$ pwd
/Users/taewan
$ ls ~/anaconda3
Anaconda-Navigator.app             phrasebooks
bin                                pkgs
conda-meta                         plugins
doc                                python.app
etc                                qml
include                            resources
lib                                sbin
libexec                            share
man                                ssl
mkspecs                            translations
org.freedesktop.dbus-session.plist
$ echo 'export PATH="/Users/taewan/anaconda3/bin:$PATH"' >> ~/.bashrc
$ echo 'export PATH="/Users/taewan/anaconda3/bin:$PATH"' >> ~/.zshrc
$ source ~/.bashrc
$ source ~/.zshrc
$
```

환경 변수가 변경되면 다음과 같은 명령으로 Anaconda 설치 상태를 확인합니다.


```
$ conda -V
conda 4.5.12
$
```



## 3. Anaconda 가상환경 구성
다음 명령을 실행하며 pytorch_env 가상환경을 만들 수 있습니다.


```
~/pytorch $ conda create --name pytorch_env python=3
## 중간 로그 생략

The following NEW packages will be INSTALLED:

    ca-certificates: 2019.1.23-0
    certifi:         2018.11.29-py37_0
    libcxx:          4.0.1-hcfea43d_1
    libcxxabi:       4.0.1-hcfea43d_1
    libedit:         3.1.20181209-hb402a30_0
    libffi:          3.2.1-h475c297_4
    ncurses:         6.1-h0a44026_1
    openssl:         1.1.1a-h1de35cc_0
    pip:             19.0.1-py37_0
    python:          3.7.2-haf84260_0
    readline:        7.0-h1de35cc_5
    setuptools:      40.8.0-py37_0
    sqlite:          3.26.0-ha441bb4_0
    tk:              8.6.8-ha441bb4_0
    wheel:           0.32.3-py37_0
    xz:              5.2.4-h1de35cc_4
    zlib:            1.2.11-h1de35cc_3

## 설치 확인 문의 및 동의 ==> y입력
Proceed ([y]/n)? y

#
# To activate this environment, use:
# > source activate pytorch_env
#
# To deactivate an active environment, use:
# > source deactivate
#
~/pytorch $
```



## 4. Anaconda 가상환경 활성화 및 패키지 설치
다음 명령을 실행하여 앞에서 생성한 pytorch_env를 활성화 시킬 수 있습니다.

* torch
* torchvision
* torchtext
* scikit-learn
* matplotlib


```
~/pytorch $ source activate pytorch_env
(pytorch_env) ~/pytorch $ conda install -y pytorch-cpu torchvision-cpu -c pytorch
## 로그 생략

(pytorch_env) ~/pytorch $ conda install -y -c derickl torchtext
## 로그 생략

(pytorch_env) ~/pytorch $ conda install -y scikit-learn
## 로그 생략

(pytorch_env) ~/pytorch $ conda install -y matplotlib
## 로그 생략

(pytorch_env) ~/pytorch $ conda install -y pandas
## 로그 생략

##가상환경 종료
(pytorch_env) ~/pytorch $ deactivate 

~/pytorch $
```

```
!pytorch, torchtext가 install되지 않는다면 다음과같이 수행하면 된다.
onda install abipy -c abinit

/* on Mac and Linux and python 3.6.6 and the installation completed successfully
(windows is not supported)
Did you add conda-forge and matsci to the default channels with */

conda config --add channels conda-forge
conda config --add channels matsci
conda config --add channels abinit

/* If this does not solve your problem, you may try inside your conda environment. */
pip install abipy

```




## 5. Jupyter 실습 환경 구성
지금까지 파이썬과 실습에 필요한 라이브러리를 모두 설치했습니다. 이번 절에서는 실습 코드를 내려받고 Jupyter Notebook을 실행하는 방법에 대하여 알아보겠습니다.
실습 코드는 github에서 제공됩니다.

github repository : https://github.com/gkqlsdlek123/pytorch/
(PyTorch 1.0 지원)



### 5.1 Jupyter Notebook 실행
다음 명령을 입력하여 Anaconda 가상환경을 활성화 시키고 Github 레파지토리 최상위 디렉터리에서 Jupyter Notebook을 실행합니다.
```
 ~/habin_kim/pytorch > source activate pytorch_env
(pytorch_env)  ~/habin_kim/pytorch > jupyter notebook

[I 16:41:54.001 NotebookApp] Serving notebooks from local directory: /Users/habin_kim/pytorch
[I 16:41:54.001 NotebookApp] The Jupyter Notebook is running at:
[I 16:41:54.001 NotebookApp] http://localhost:8888/?token=ffb71cb8ed2b9bf909404fa314074244d394ad97dfd173b6
[I 16:41:54.001 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[C 16:41:54.010 NotebookApp]
```
    To access the notebook, open this file in a browser:
        file:///Users/habin_kim/Library/Jupyter/runtime/nbserver-49921-open.html
    Or copy and paste one of these URLs:
        http://localhost:8787/?token=ffb71cb8ed2b9bf909404fa314074244d394ad97dfd173b6

위 명령을 입력하면 http://localhost:8787/tree 주소로 기본 브라우저가 실행됩니다. 원격 서버일 경우 위 실행 로그에 출력된 URL로 웹 페이지를 오픈하면 됩니다.

http://localhost:8888/?token=ffb71cb8ed2b9bf909404fa314074244d394ad97dfd173b6
