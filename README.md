# [HATE-SPEECH-DETECTION]()  ⚡️ [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/jhabarsingh/GRAPH-TESTCASE-VISUALIZER/blob/main/LICENSE) [![GitHub stars](https://img.shields.io/github/stars/jhabarsingh/HATE-SPEECH-DETECTION)](https://github.com/jhabarsingh/HATE-SPEECH-DETECTION/stargazers)  [![GitHub contributors](https://img.shields.io/github/contributors/jhabarsingh/HATE-SPEECH-DETECTION.svg)](https://github.com/jhabarsingh/HATE-SPEECH-DETECTION/graphs/contributors)  [![GitHub issues](https://img.shields.io/github/issues/jhabarsingh/HATE-SPEECH-DETECTION.svg)](https://github.com/jhabarsingh/HATE-SPEECH-DETECTION/issues) [![GitHub forks](https://img.shields.io/github/forks/jhabarsingh/HATE-SPEECH-DETECTION.svg?style=social&label=Fork)](https://GitHub.com/jhabarsingh/HATE-SPEECH-DETECTION/network/)

<p align="center">
  <img src="https://github.com/jhabarsingh/HATE-SPEECH-DETECTION/blob/main/doc/org_logo.png?raw=true" />
</p>
<details>
  <summary>:zap: TECH STACK</summary>
  <br/>
  <div style="display:flex;justify-content:space-around">
  <img  title="Django" src="https://icon-library.com/images/django-icon/django-icon-0.jpg" width="50px" height="50px" style="margin-right:5px;" />
  <img titlt="Vuejs"   src="https://vuejs.org/images/logo.png" width="50px" height="50px"  style="margin-right:5px;"/>
  <img  title="Vuex" src="https://s3.amazonaws.com/coursetro/posts/144-full.png"  height="50px" style="margin-right:5px;"     />
  <img  title="Vuetify" src="https://cdn.worldvectorlogo.com/logos/vuetify.svg" height="50px"  style="margin-right:5px;"/>
    <img  title="Tensorflow" src="https://www.altoros.com/blog/wp-content/uploads/2016/01/tensorflow-logo-cropped.png" height="50px" style="margin-right:5px;" />
  
  <img  title="Docker" src="https://pbs.twimg.com/profile_images/1273307847103635465/lfVWBmiW_400x400.png" height="50px" style="margin-right:5px;" />
</div>
</details>



## About
  [HATE-SPEECH-DETECTION]() is a Web app built using **Django** and **Vuejs**. Uses **Recurrent neural network LSTM** algorithm to detect **hatespeech** in a text. Hate speech detector can be used as a middleware between the servers and a client. Now a days we see many hatefull comments on social sites so having a detector like this can stop the hate spread to a large extend.

## Preview

![Preview](https://github.com/jhabarsingh/HATE-SPEECH-DETECTION/blob/main/doc/preview.gif?raw=true)



## Django Backend Setup
![Django Apis](https://github.com/jhabarsingh/HATE-SPEECH-DETECTION/blob/main/doc/apis.png?raw=true)

### Using venv
```bash
git clone https://github.com/jhabarsingh/HATE-SPEECH-DETECTION.git 
cd HATE-SPEECH-DETECTION
python3 -m venv env # Python 3.6.9 or 3.7.0 version 
source env/bin/activate
python3 -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py runserver
```

### Using conda
```bash
git clone https://github.com/jhabarsingh/HATE-SPEECH-DETECTION.git 
cd HATE-SPEECH-DETECTION
conda create -n hatespeech python==3.7 
conda activate hatespeech
python3 -m pip install --upgrade pip
pip install -r requirements.txt
python manage.py runserver
```

### Using Docker

```bash
sudo apt-get install docker docker-compose # Install docker, docker-compose on linux
git clone https://github.com/jhabarsingh/HATE-SPEECH-DETECTION.git
cd HATE-SPEECH-DETECTION
sudo docker-compose up
```

## Vuejs Setup

[Install node](https://nodejs.org/en/download/) |
[Install vue/cli](https://cli.vuejs.org/)
```bash
 git clone https://github.com/jhabarsingh/HATE-SPEECH-DETECTION.git
 cd HATE-SPEECH-DETECTION
 npm install   # Nodejs should be installed
 npm run start # Vuejs should be installed
 ```


# [Want To Contribute](https://medium.com/mindsdb/contributing-to-an-open-source-project-how-to-get-started-6ba812301738)
### You can contribute to this project in many ways
 1. You can create an issue if you find any bug.
 2. You can work on an existing issue and Send PR.
 3. You can make changes in the design if it is needed.
 4. Even if you find any grammatical or spelling mistakes then also you can create an issue.

> *I would be glad to see a notification saying `User {xyz} created a Pull Request`.
I promise to review it.*
