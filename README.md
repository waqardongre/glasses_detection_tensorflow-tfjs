# Real-time Glasses Detection with Tensorflow-tfjs


Reference:
https://blog.tensorflow.org/2021/01/custom-object-detection-in-browser.html Customized as per our need.

Connecting to Google Drive
from google.colab import drive
drive.mount('/content/drive')
Mounted at /content/drive
path_root = "/content/drive/MyDrive/DSs/glasses_detection_project"
!cd {path_root}/
Training the model
With a good dataset, it’s time to think about the model.TensorFlow 2 provides an Object Detection API that makes it easy to construct, train, and deploy object detection models. In this project, we’re going to use this API and train the model using a Google Colaboratory Notebook. The remainder of this section explains how to set up the environment, the model selection, and training. If you want to jump straight to the Colab Notebook, click here.

Setting up the environment
Create a new Google Colab notebook and select a GPU as hardware accelerator:

Runtime > Change runtime type > Hardware accelerator: GPU

Clone and install the Tensorflow Object Detection API
In order to use the TensorFlow Object Detection API, we need to clone it's GitHub Repo.

Dependencies
Most of the dependencies required come preloaded in Google Colab. No extra installation is needed.

Protocol Buffers
The TensorFlow Object Detection API relies on what are called protocol buffers (also known as protobufs). Protobufs are a language neutral way to describe information. That means you can write a protobuf once and then compile it to be used with other languages, like Python, Java or C [5].

The protoc command used below is compiling all the protocol buffers in the object_detection/protos folder for Python.

Run once if using google drive or once per session
!git clone https://github.com/tensorflow/models.git {path_root}/models/
Cloning into '/content/drive/MyDrive/DSs/glasses_detection_project/models'...
remote: Enumerating objects: 74696, done.
remote: Counting objects: 100% (184/184), done.
remote: Compressing objects: 100% (99/99), done.
remote: Total 74696 (delta 97), reused 157 (delta 82), pack-reused 74512
Receiving objects: 100% (74696/74696), 580.36 MiB | 15.75 MiB/s, done.
Resolving deltas: 100% (52967/52967), done.
Checking out files: 100% (3080/3080), done.
Installing protobufs
%cd {path_root}/models/research/
!protoc object_detection/protos/*.proto --python_out=.
# Install TensorFlow Object Detection API.
!cp object_detection/packages/tf2/setup.py .
!python -m pip install .
/content/drive/.shortcut-targets-by-id/1ICDl8YeYQgBB5FK2XWH-e75qELCDjzqZ/DSs/glasses_detection_project/models/research
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Processing /content/drive/.shortcut-targets-by-id/1ICDl8YeYQgBB5FK2XWH-e75qELCDjzqZ/DSs/glasses_detection_project/models/research
  DEPRECATION: A future pip version will change local packages to be built in-place without first copying to a temporary directory. We recommend you use --use-feature=in-tree-build to test your packages with this new behavior before it becomes the default.
   pip 21.3 will remove support for this functionality. You can find discussion regarding this at https://github.com/pypa/pip/issues/7555.
Collecting avro-python3
  Downloading avro-python3-1.10.2.tar.gz (38 kB)
Collecting apache-beam
  Downloading apache_beam-2.40.0-cp37-cp37m-manylinux2010_x86_64.whl (10.9 MB)
     |████████████████████████████████| 10.9 MB 11.3 MB/s 
Requirement already satisfied: pillow in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (7.1.2)
Requirement already satisfied: lxml in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (4.2.6)
Requirement already satisfied: matplotlib in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (3.2.2)
Requirement already satisfied: Cython in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (0.29.30)
Requirement already satisfied: contextlib2 in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (0.5.5)
Collecting tf-slim
  Downloading tf_slim-1.1.0-py2.py3-none-any.whl (352 kB)
     |████████████████████████████████| 352 kB 77.0 MB/s 
Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (1.15.0)
Requirement already satisfied: pycocotools in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (2.0.4)
Collecting lvis
  Downloading lvis-0.5.3-py3-none-any.whl (14 kB)
Requirement already satisfied: scipy in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (1.4.1)
Requirement already satisfied: pandas in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (1.3.5)
Collecting tf-models-official>=2.5.1
  Downloading tf_models_official-2.9.2-py2.py3-none-any.whl (2.1 MB)
     |████████████████████████████████| 2.1 MB 42.0 MB/s 
Collecting tensorflow_io
  Downloading tensorflow_io-0.26.0-cp37-cp37m-manylinux_2_12_x86_64.manylinux2010_x86_64.whl (25.9 MB)
     |████████████████████████████████| 25.9 MB 1.3 MB/s 
Requirement already satisfied: keras in /usr/local/lib/python3.7/dist-packages (from object-detection==0.1) (2.7.0)
Collecting pyparsing==2.4.7
  Downloading pyparsing-2.4.7-py2.py3-none-any.whl (67 kB)
     |████████████████████████████████| 67 kB 6.7 MB/s 
Requirement already satisfied: gin-config in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (0.5.0)
Collecting sacrebleu
  Downloading sacrebleu-2.1.0-py3-none-any.whl (92 kB)
     |████████████████████████████████| 92 kB 14.4 MB/s 
Requirement already satisfied: oauth2client in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (4.1.3)
Requirement already satisfied: tensorflow-hub>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (0.12.0)
Requirement already satisfied: opencv-python-headless in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (4.5.2.52)
Collecting tensorflow-text~=2.9.0
  Downloading tensorflow_text-2.9.0-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (4.6 MB)
     |████████████████████████████████| 4.6 MB 40.2 MB/s 
Collecting py-cpuinfo>=3.3.0
  Downloading py-cpuinfo-8.0.0.tar.gz (99 kB)
     |████████████████████████████████| 99 kB 11.8 MB/s 
Collecting sentencepiece
  Downloading sentencepiece-0.1.96-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.2 MB)
     |████████████████████████████████| 1.2 MB 54.6 MB/s 
Collecting pyyaml<6.0,>=5.1
  Downloading PyYAML-5.4.1-cp37-cp37m-manylinux1_x86_64.whl (636 kB)
     |████████████████████████████████| 636 kB 73.5 MB/s 
Collecting tensorflow-model-optimization>=0.4.1
  Downloading tensorflow_model_optimization-0.7.2-py2.py3-none-any.whl (237 kB)
     |████████████████████████████████| 237 kB 75.5 MB/s 
Requirement already satisfied: kaggle>=1.3.9 in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (1.5.12)
Collecting tensorflow-addons
  Downloading tensorflow_addons-0.17.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
     |████████████████████████████████| 1.1 MB 62.3 MB/s 
Requirement already satisfied: google-api-python-client>=1.6.7 in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (1.12.11)
Requirement already satisfied: numpy>=1.20 in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (1.21.6)
Requirement already satisfied: psutil>=5.4.3 in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (5.4.8)
Requirement already satisfied: tensorflow-datasets in /usr/local/lib/python3.7/dist-packages (from tf-models-official>=2.5.1->object-detection==0.1) (4.0.1)
Collecting tensorflow~=2.9.0
  Downloading tensorflow-2.9.1-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (511.7 MB)
     |████████████████████████████████| 511.7 MB 6.3 kB/s 
Collecting seqeval
  Downloading seqeval-1.2.2.tar.gz (43 kB)
     |████████████████████████████████| 43 kB 2.3 MB/s 
Requirement already satisfied: httplib2<1dev,>=0.15.0 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (0.17.4)
Requirement already satisfied: uritemplate<4dev,>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (3.0.1)
Requirement already satisfied: google-api-core<3dev,>=1.21.0 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (1.31.6)
Requirement already satisfied: google-auth<3dev,>=1.16.0 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (1.35.0)
Requirement already satisfied: google-auth-httplib2>=0.0.3 in /usr/local/lib/python3.7/dist-packages (from google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (0.0.4)
Requirement already satisfied: requests<3.0.0dev,>=2.18.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (2.23.0)
Requirement already satisfied: pytz in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (2022.1)
Requirement already satisfied: googleapis-common-protos<2.0dev,>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (1.56.2)
Requirement already satisfied: setuptools>=40.3.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (57.4.0)
Requirement already satisfied: packaging>=14.3 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (21.3)
Requirement already satisfied: protobuf<4.0.0dev,>=3.12.0 in /usr/local/lib/python3.7/dist-packages (from google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (3.17.3)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (4.2.4)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (0.2.8)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (4.8)
Requirement already satisfied: python-slugify in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official>=2.5.1->object-detection==0.1) (6.1.2)
Requirement already satisfied: python-dateutil in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official>=2.5.1->object-detection==0.1) (2.8.2)
Requirement already satisfied: certifi in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official>=2.5.1->object-detection==0.1) (2022.6.15)
Requirement already satisfied: tqdm in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official>=2.5.1->object-detection==0.1) (4.64.0)
Requirement already satisfied: urllib3 in /usr/local/lib/python3.7/dist-packages (from kaggle>=1.3.9->tf-models-official>=2.5.1->object-detection==0.1) (1.24.3)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3dev,>=1.16.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (0.4.8)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (2.10)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (3.0.4)
Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.1.2)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.46.3)
Collecting keras
  Downloading keras-2.9.0-py2.py3-none-any.whl (1.6 MB)
     |████████████████████████████████| 1.6 MB 53.9 MB/s 
Collecting flatbuffers<2,>=1.12
  Downloading flatbuffers-1.12-py2.py3-none-any.whl (15 kB)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.14.1)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (0.2.0)
Requirement already satisfied: libclang>=13.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (14.0.1)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (4.1.1)
Requirement already satisfied: gast<=0.4.0,>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (0.4.0)
Collecting tensorflow-estimator<2.10.0,>=2.9.0rc0
  Downloading tensorflow_estimator-2.9.0-py2.py3-none-any.whl (438 kB)
     |████████████████████████████████| 438 kB 77.7 MB/s 
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.6.3)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (3.3.0)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (0.26.0)
Collecting tensorboard<2.10,>=2.9
  Downloading tensorboard-2.9.1-py3-none-any.whl (5.8 MB)
     |████████████████████████████████| 5.8 MB 52.7 MB/s 
Requirement already satisfied: absl-py>=1.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.1.0)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.1.0)
Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (3.1.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.7/dist-packages (from astunparse>=1.6.0->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (0.37.1)
Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.5.2)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (0.6.1)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (3.3.7)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.0.1)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.8.1)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (0.4.6)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (1.3.1)
Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (4.11.4)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (3.8.0)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.10,>=2.9->tensorflow~=2.9.0->tf-models-official>=2.5.1->object-detection==0.1) (3.2.0)
Requirement already satisfied: dm-tree~=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow-model-optimization>=0.4.1->tf-models-official>=2.5.1->object-detection==0.1) (0.1.7)
Requirement already satisfied: pyarrow<8.0.0,>=0.15.1 in /usr/local/lib/python3.7/dist-packages (from apache-beam->object-detection==0.1) (6.0.1)
Collecting hdfs<3.0.0,>=2.1.0
  Downloading hdfs-2.7.0-py3-none-any.whl (34 kB)
Requirement already satisfied: crcmod<2.0,>=1.7 in /usr/local/lib/python3.7/dist-packages (from apache-beam->object-detection==0.1) (1.7)
Collecting pymongo<4.0.0,>=3.8.0
  Downloading pymongo-3.12.3-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (508 kB)
     |████████████████████████████████| 508 kB 73.8 MB/s 
Requirement already satisfied: pydot<2,>=1.2.0 in /usr/local/lib/python3.7/dist-packages (from apache-beam->object-detection==0.1) (1.3.0)
Collecting requests<3.0.0dev,>=2.18.0
  Downloading requests-2.28.0-py3-none-any.whl (62 kB)
     |████████████████████████████████| 62 kB 1.4 MB/s 
Collecting proto-plus<2,>=1.7.1
  Downloading proto_plus-1.20.6-py3-none-any.whl (46 kB)
     |████████████████████████████████| 46 kB 5.2 MB/s 
Collecting fastavro<2,>=0.23.6
  Downloading fastavro-1.5.2-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (2.3 MB)
     |████████████████████████████████| 2.3 MB 57.3 MB/s 
Collecting orjson<4.0
  Downloading orjson-3.7.5-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (272 kB)
     |████████████████████████████████| 272 kB 72.8 MB/s 
Collecting dill<0.3.2,>=0.3.1.1
  Downloading dill-0.3.1.1.tar.gz (151 kB)
     |████████████████████████████████| 151 kB 76.8 MB/s 
Collecting cloudpickle<3,>=2.1.0
  Downloading cloudpickle-2.1.0-py3-none-any.whl (25 kB)
Requirement already satisfied: docopt in /usr/local/lib/python3.7/dist-packages (from hdfs<3.0.0,>=2.1.0->apache-beam->object-detection==0.1) (0.6.2)
Collecting protobuf<4.0.0dev,>=3.12.0
  Downloading protobuf-3.19.4-cp37-cp37m-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (1.1 MB)
     |████████████████████████████████| 1.1 MB 64.8 MB/s 
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.7/dist-packages (from requests<3.0.0dev,>=2.18.0->google-api-core<3dev,>=1.21.0->google-api-python-client>=1.6.7->tf-models-official>=2.5.1->object-detection==0.1) (2.0.12)
Requirement already satisfied: opencv-python>=4.1.0.25 in /usr/local/lib/python3.7/dist-packages (from lvis->object-detection==0.1) (4.1.2.30)
Requirement already satisfied: kiwisolver>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from lvis->object-detection==0.1) (1.4.3)
Requirement already satisfied: cycler>=0.10.0 in /usr/local/lib/python3.7/dist-packages (from lvis->object-detection==0.1) (0.11.0)
Requirement already satisfied: text-unidecode>=1.3 in /usr/local/lib/python3.7/dist-packages (from python-slugify->kaggle>=1.3.9->tf-models-official>=2.5.1->object-detection==0.1) (1.3)
Collecting portalocker
  Downloading portalocker-2.4.0-py2.py3-none-any.whl (16 kB)
Requirement already satisfied: tabulate>=0.8.9 in /usr/local/lib/python3.7/dist-packages (from sacrebleu->tf-models-official>=2.5.1->object-detection==0.1) (0.8.9)
Requirement already satisfied: regex in /usr/local/lib/python3.7/dist-packages (from sacrebleu->tf-models-official>=2.5.1->object-detection==0.1) (2022.6.2)
Collecting colorama
  Downloading colorama-0.4.5-py2.py3-none-any.whl (16 kB)
Requirement already satisfied: scikit-learn>=0.21.3 in /usr/local/lib/python3.7/dist-packages (from seqeval->tf-models-official>=2.5.1->object-detection==0.1) (1.0.2)
Requirement already satisfied: joblib>=0.11 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.21.3->seqeval->tf-models-official>=2.5.1->object-detection==0.1) (1.1.0)
Requirement already satisfied: threadpoolctl>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from scikit-learn>=0.21.3->seqeval->tf-models-official>=2.5.1->object-detection==0.1) (3.1.0)
Requirement already satisfied: typeguard>=2.7 in /usr/local/lib/python3.7/dist-packages (from tensorflow-addons->tf-models-official>=2.5.1->object-detection==0.1) (2.7.1)
Requirement already satisfied: promise in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official>=2.5.1->object-detection==0.1) (2.3)
Requirement already satisfied: attrs>=18.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official>=2.5.1->object-detection==0.1) (21.4.0)
Requirement already satisfied: tensorflow-metadata in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official>=2.5.1->object-detection==0.1) (1.8.0)
Requirement already satisfied: importlib-resources in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official>=2.5.1->object-detection==0.1) (5.7.1)
Requirement already satisfied: future in /usr/local/lib/python3.7/dist-packages (from tensorflow-datasets->tf-models-official>=2.5.1->object-detection==0.1) (0.16.0)
Building wheels for collected packages: object-detection, py-cpuinfo, dill, avro-python3, seqeval
  Building wheel for object-detection (setup.py) ... done
  Created wheel for object-detection: filename=object_detection-0.1-py3-none-any.whl size=1694695 sha256=6fb026eca2ce19952e12efa5583be75f4b90c7c1a7f8888b4925f21bd981b07c
  Stored in directory: /tmp/pip-ephem-wheel-cache-jomgt9h5/wheels/fe/47/fd/558545d5063d535d9f2d9bccdad38769869fda8c9712d17549
  Building wheel for py-cpuinfo (setup.py) ... done
  Created wheel for py-cpuinfo: filename=py_cpuinfo-8.0.0-py3-none-any.whl size=22257 sha256=9bb208a0932fcea4ddb96ac8a5f05a1d279175523aa8c86973270bfad8dc5130
  Stored in directory: /root/.cache/pip/wheels/d2/f1/1f/041add21dc9c4220157f1bd2bd6afe1f1a49524c3396b94401
  Building wheel for dill (setup.py) ... done
  Created wheel for dill: filename=dill-0.3.1.1-py3-none-any.whl size=78544 sha256=46b5b0ddd1a68989ab2079b8e695a595ff9f1f9f4d761f8cecde1d44edfb44e2
  Stored in directory: /root/.cache/pip/wheels/a4/61/fd/c57e374e580aa78a45ed78d5859b3a44436af17e22ca53284f
  Building wheel for avro-python3 (setup.py) ... done
  Created wheel for avro-python3: filename=avro_python3-1.10.2-py3-none-any.whl size=44010 sha256=af4457495250b007df5e070d9272eed2972c037d86cb4b1bcc9eebf9f48279d0
  Stored in directory: /root/.cache/pip/wheels/d6/e5/b1/6b151d9b535ee50aaa6ab27d145a0104b6df02e5636f0376da
  Building wheel for seqeval (setup.py) ... done
  Created wheel for seqeval: filename=seqeval-1.2.2-py3-none-any.whl size=16180 sha256=23bb6f19724295fc51ce4e102399424a45cf74dbb1e36122d4119f4cf401f59c
  Stored in directory: /root/.cache/pip/wheels/05/96/ee/7cac4e74f3b19e3158dce26a20a1c86b3533c43ec72a549fd7
Successfully built object-detection py-cpuinfo dill avro-python3 seqeval
Installing collected packages: requests, pyparsing, protobuf, tensorflow-estimator, tensorboard, keras, flatbuffers, tensorflow, portalocker, dill, colorama, tf-slim, tensorflow-text, tensorflow-model-optimization, tensorflow-addons, seqeval, sentencepiece, sacrebleu, pyyaml, pymongo, py-cpuinfo, proto-plus, orjson, hdfs, fastavro, cloudpickle, tf-models-official, tensorflow-io, lvis, avro-python3, apache-beam, object-detection
  Attempting uninstall: requests
    Found existing installation: requests 2.23.0
    Uninstalling requests-2.23.0:
      Successfully uninstalled requests-2.23.0
  Attempting uninstall: pyparsing
    Found existing installation: pyparsing 3.0.9
    Uninstalling pyparsing-3.0.9:
      Successfully uninstalled pyparsing-3.0.9
  Attempting uninstall: protobuf
    Found existing installation: protobuf 3.17.3
    Uninstalling protobuf-3.17.3:
      Successfully uninstalled protobuf-3.17.3
  Attempting uninstall: tensorflow-estimator
    Found existing installation: tensorflow-estimator 2.7.0
    Uninstalling tensorflow-estimator-2.7.0:
      Successfully uninstalled tensorflow-estimator-2.7.0
  Attempting uninstall: tensorboard
    Found existing installation: tensorboard 2.8.0
    Uninstalling tensorboard-2.8.0:
      Successfully uninstalled tensorboard-2.8.0
  Attempting uninstall: keras
    Found existing installation: keras 2.7.0
    Uninstalling keras-2.7.0:
      Successfully uninstalled keras-2.7.0
  Attempting uninstall: flatbuffers
    Found existing installation: flatbuffers 2.0
    Uninstalling flatbuffers-2.0:
      Successfully uninstalled flatbuffers-2.0
  Attempting uninstall: tensorflow
    Found existing installation: tensorflow 2.7.0+zzzcolab20220506150900
    Uninstalling tensorflow-2.7.0+zzzcolab20220506150900:
      Successfully uninstalled tensorflow-2.7.0+zzzcolab20220506150900
  Attempting uninstall: dill
    Found existing installation: dill 0.3.5.1
    Uninstalling dill-0.3.5.1:
      Successfully uninstalled dill-0.3.5.1
  Attempting uninstall: pyyaml
    Found existing installation: PyYAML 3.13
    Uninstalling PyYAML-3.13:
      Successfully uninstalled PyYAML-3.13
  Attempting uninstall: pymongo
    Found existing installation: pymongo 4.1.1
    Uninstalling pymongo-4.1.1:
      Successfully uninstalled pymongo-4.1.1
  Attempting uninstall: cloudpickle
    Found existing installation: cloudpickle 1.3.0
    Uninstalling cloudpickle-1.3.0:
      Successfully uninstalled cloudpickle-1.3.0
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
multiprocess 0.70.13 requires dill>=0.3.5.1, but you have dill 0.3.1.1 which is incompatible.
gym 0.17.3 requires cloudpickle<1.7.0,>=1.2.0, but you have cloudpickle 2.1.0 which is incompatible.
google-colab 1.0.0 requires requests~=2.23.0, but you have requests 2.28.0 which is incompatible.
datascience 0.10.6 requires folium==0.2.1, but you have folium 0.8.3 which is incompatible.
albumentations 0.1.12 requires imgaug<0.2.7,>=0.2.5, but you have imgaug 0.2.9 which is incompatible.
Successfully installed apache-beam-2.40.0 avro-python3-1.10.2 cloudpickle-2.1.0 colorama-0.4.5 dill-0.3.1.1 fastavro-1.5.2 flatbuffers-1.12 hdfs-2.7.0 keras-2.9.0 lvis-0.5.3 object-detection-0.1 orjson-3.7.5 portalocker-2.4.0 proto-plus-1.20.6 protobuf-3.19.4 py-cpuinfo-8.0.0 pymongo-3.12.3 pyparsing-2.4.7 pyyaml-5.4.1 requests-2.28.0 sacrebleu-2.1.0 sentencepiece-0.1.96 seqeval-1.2.2 tensorboard-2.9.1 tensorflow-2.9.1 tensorflow-addons-0.17.1 tensorflow-estimator-2.9.0 tensorflow-io-0.26.0 tensorflow-model-optimization-0.7.2 tensorflow-text-2.9.0 tf-models-official-2.9.2 tf-slim-1.1.0
Run the model builder test - Optional to run
!python {path_root}/models/research/object_detection/builders/model_builder_tf2_test.py
Running tests under Python 3.7.13: /usr/bin/python3
[ RUN      ] ModelBuilderTF2Test.test_create_center_net_deepmac
2022-06-28 08:21:07.852227: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:42] Overriding orig_value setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
W0628 08:21:08.099057 140347812091776 model_builder.py:1102] Building experimental DeepMAC meta-arch. Some features may be omitted.
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_center_net_deepmac): 1.5s
I0628 08:21:08.461358 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_center_net_deepmac): 1.5s
[       OK ] ModelBuilderTF2Test.test_create_center_net_deepmac
[ RUN      ] ModelBuilderTF2Test.test_create_center_net_model0 (customize_head_params=True)
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_center_net_model0 (customize_head_params=True)): 0.53s
I0628 08:21:08.991438 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_center_net_model0 (customize_head_params=True)): 0.53s
[       OK ] ModelBuilderTF2Test.test_create_center_net_model0 (customize_head_params=True)
[ RUN      ] ModelBuilderTF2Test.test_create_center_net_model1 (customize_head_params=False)
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_center_net_model1 (customize_head_params=False)): 0.26s
I0628 08:21:09.254272 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_center_net_model1 (customize_head_params=False)): 0.26s
[       OK ] ModelBuilderTF2Test.test_create_center_net_model1 (customize_head_params=False)
[ RUN      ] ModelBuilderTF2Test.test_create_center_net_model_from_keypoints
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_center_net_model_from_keypoints): 0.24s
I0628 08:21:09.495411 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_center_net_model_from_keypoints): 0.24s
[       OK ] ModelBuilderTF2Test.test_create_center_net_model_from_keypoints
[ RUN      ] ModelBuilderTF2Test.test_create_center_net_model_mobilenet
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_center_net_model_mobilenet): 1.81s
I0628 08:21:11.309516 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_center_net_model_mobilenet): 1.81s
[       OK ] ModelBuilderTF2Test.test_create_center_net_model_mobilenet
[ RUN      ] ModelBuilderTF2Test.test_create_experimental_model
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_experimental_model): 0.0s
I0628 08:21:11.310434 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_experimental_model): 0.0s
[       OK ] ModelBuilderTF2Test.test_create_experimental_model
[ RUN      ] ModelBuilderTF2Test.test_create_faster_rcnn_from_config_with_crop_feature0 (True)
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_from_config_with_crop_feature0 (True)): 0.02s
I0628 08:21:11.333110 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_from_config_with_crop_feature0 (True)): 0.02s
[       OK ] ModelBuilderTF2Test.test_create_faster_rcnn_from_config_with_crop_feature0 (True)
[ RUN      ] ModelBuilderTF2Test.test_create_faster_rcnn_from_config_with_crop_feature1 (False)
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_from_config_with_crop_feature1 (False)): 0.01s
I0628 08:21:11.346976 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_from_config_with_crop_feature1 (False)): 0.01s
[       OK ] ModelBuilderTF2Test.test_create_faster_rcnn_from_config_with_crop_feature1 (False)
[ RUN      ] ModelBuilderTF2Test.test_create_faster_rcnn_model_from_config_with_example_miner
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_model_from_config_with_example_miner): 0.01s
I0628 08:21:11.360754 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_model_from_config_with_example_miner): 0.01s
[       OK ] ModelBuilderTF2Test.test_create_faster_rcnn_model_from_config_with_example_miner
[ RUN      ] ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul): 0.09s
I0628 08:21:11.452283 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul): 0.09s
[       OK ] ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_faster_rcnn_with_matmul
[ RUN      ] ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul): 0.09s
I0628 08:21:11.542032 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul): 0.09s
[       OK ] ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_faster_rcnn_without_matmul
[ RUN      ] ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul): 0.1s
I0628 08:21:11.638693 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul): 0.1s
[       OK ] ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_mask_rcnn_with_matmul
[ RUN      ] ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul): 0.09s
I0628 08:21:11.730547 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul): 0.09s
[       OK ] ModelBuilderTF2Test.test_create_faster_rcnn_models_from_config_mask_rcnn_without_matmul
[ RUN      ] ModelBuilderTF2Test.test_create_rfcn_model_from_config
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_rfcn_model_from_config): 0.09s
I0628 08:21:11.817896 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_rfcn_model_from_config): 0.09s
[       OK ] ModelBuilderTF2Test.test_create_rfcn_model_from_config
[ RUN      ] ModelBuilderTF2Test.test_create_ssd_fpn_model_from_config
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_ssd_fpn_model_from_config): 0.03s
I0628 08:21:11.843911 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_ssd_fpn_model_from_config): 0.03s
[       OK ] ModelBuilderTF2Test.test_create_ssd_fpn_model_from_config
[ RUN      ] ModelBuilderTF2Test.test_create_ssd_models_from_config
I0628 08:21:12.025502 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:146] EfficientDet EfficientNet backbone version: efficientnet-b0
I0628 08:21:12.025802 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 64
I0628 08:21:12.025939 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:149] EfficientDet BiFPN num iterations: 3
I0628 08:21:12.028331 140347812091776 efficientnet_model.py:143] round_filter input=32 output=32
I0628 08:21:12.044355 140347812091776 efficientnet_model.py:143] round_filter input=32 output=32
I0628 08:21:12.044456 140347812091776 efficientnet_model.py:143] round_filter input=16 output=16
I0628 08:21:12.100875 140347812091776 efficientnet_model.py:143] round_filter input=16 output=16
I0628 08:21:12.100978 140347812091776 efficientnet_model.py:143] round_filter input=24 output=24
I0628 08:21:12.247995 140347812091776 efficientnet_model.py:143] round_filter input=24 output=24
I0628 08:21:12.248111 140347812091776 efficientnet_model.py:143] round_filter input=40 output=40
I0628 08:21:12.413855 140347812091776 efficientnet_model.py:143] round_filter input=40 output=40
I0628 08:21:12.414013 140347812091776 efficientnet_model.py:143] round_filter input=80 output=80
I0628 08:21:12.630924 140347812091776 efficientnet_model.py:143] round_filter input=80 output=80
I0628 08:21:12.631084 140347812091776 efficientnet_model.py:143] round_filter input=112 output=112
I0628 08:21:12.865570 140347812091776 efficientnet_model.py:143] round_filter input=112 output=112
I0628 08:21:12.865734 140347812091776 efficientnet_model.py:143] round_filter input=192 output=192
I0628 08:21:13.177201 140347812091776 efficientnet_model.py:143] round_filter input=192 output=192
I0628 08:21:13.177357 140347812091776 efficientnet_model.py:143] round_filter input=320 output=320
I0628 08:21:13.253263 140347812091776 efficientnet_model.py:143] round_filter input=1280 output=1280
I0628 08:21:13.283999 140347812091776 efficientnet_model.py:453] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.0, resolution=224, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
I0628 08:21:13.480046 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:146] EfficientDet EfficientNet backbone version: efficientnet-b1
I0628 08:21:13.480213 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 88
I0628 08:21:13.480288 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:149] EfficientDet BiFPN num iterations: 4
I0628 08:21:13.481794 140347812091776 efficientnet_model.py:143] round_filter input=32 output=32
I0628 08:21:13.497678 140347812091776 efficientnet_model.py:143] round_filter input=32 output=32
I0628 08:21:13.497786 140347812091776 efficientnet_model.py:143] round_filter input=16 output=16
I0628 08:21:13.613068 140347812091776 efficientnet_model.py:143] round_filter input=16 output=16
I0628 08:21:13.613189 140347812091776 efficientnet_model.py:143] round_filter input=24 output=24
I0628 08:21:13.841766 140347812091776 efficientnet_model.py:143] round_filter input=24 output=24
I0628 08:21:13.841928 140347812091776 efficientnet_model.py:143] round_filter input=40 output=40
I0628 08:21:14.072794 140347812091776 efficientnet_model.py:143] round_filter input=40 output=40
I0628 08:21:14.073006 140347812091776 efficientnet_model.py:143] round_filter input=80 output=80
I0628 08:21:14.372476 140347812091776 efficientnet_model.py:143] round_filter input=80 output=80
I0628 08:21:14.372665 140347812091776 efficientnet_model.py:143] round_filter input=112 output=112
I0628 08:21:14.671498 140347812091776 efficientnet_model.py:143] round_filter input=112 output=112
I0628 08:21:14.671656 140347812091776 efficientnet_model.py:143] round_filter input=192 output=192
I0628 08:21:15.053882 140347812091776 efficientnet_model.py:143] round_filter input=192 output=192
I0628 08:21:15.054038 140347812091776 efficientnet_model.py:143] round_filter input=320 output=320
I0628 08:21:15.202463 140347812091776 efficientnet_model.py:143] round_filter input=1280 output=1280
I0628 08:21:15.231410 140347812091776 efficientnet_model.py:453] Building model efficientnet with params ModelConfig(width_coefficient=1.0, depth_coefficient=1.1, resolution=240, dropout_rate=0.2, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
I0628 08:21:15.287576 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:146] EfficientDet EfficientNet backbone version: efficientnet-b2
I0628 08:21:15.287701 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 112
I0628 08:21:15.287770 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:149] EfficientDet BiFPN num iterations: 5
I0628 08:21:15.289252 140347812091776 efficientnet_model.py:143] round_filter input=32 output=32
I0628 08:21:15.305142 140347812091776 efficientnet_model.py:143] round_filter input=32 output=32
I0628 08:21:15.305264 140347812091776 efficientnet_model.py:143] round_filter input=16 output=16
I0628 08:21:15.430741 140347812091776 efficientnet_model.py:143] round_filter input=16 output=16
I0628 08:21:15.430893 140347812091776 efficientnet_model.py:143] round_filter input=24 output=24
I0628 08:21:15.667956 140347812091776 efficientnet_model.py:143] round_filter input=24 output=24
I0628 08:21:15.668109 140347812091776 efficientnet_model.py:143] round_filter input=40 output=48
I0628 08:21:15.904365 140347812091776 efficientnet_model.py:143] round_filter input=40 output=48
I0628 08:21:15.904533 140347812091776 efficientnet_model.py:143] round_filter input=80 output=88
I0628 08:21:16.216283 140347812091776 efficientnet_model.py:143] round_filter input=80 output=88
I0628 08:21:16.216453 140347812091776 efficientnet_model.py:143] round_filter input=112 output=120
I0628 08:21:16.517625 140347812091776 efficientnet_model.py:143] round_filter input=112 output=120
I0628 08:21:16.517776 140347812091776 efficientnet_model.py:143] round_filter input=192 output=208
I0628 08:21:16.903511 140347812091776 efficientnet_model.py:143] round_filter input=192 output=208
I0628 08:21:16.903779 140347812091776 efficientnet_model.py:143] round_filter input=320 output=352
I0628 08:21:17.055815 140347812091776 efficientnet_model.py:143] round_filter input=1280 output=1408
I0628 08:21:17.089730 140347812091776 efficientnet_model.py:453] Building model efficientnet with params ModelConfig(width_coefficient=1.1, depth_coefficient=1.2, resolution=260, dropout_rate=0.3, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
I0628 08:21:17.150689 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:146] EfficientDet EfficientNet backbone version: efficientnet-b3
I0628 08:21:17.151043 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 160
I0628 08:21:17.151121 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:149] EfficientDet BiFPN num iterations: 6
I0628 08:21:17.152704 140347812091776 efficientnet_model.py:143] round_filter input=32 output=40
I0628 08:21:17.167696 140347812091776 efficientnet_model.py:143] round_filter input=32 output=40
I0628 08:21:17.167794 140347812091776 efficientnet_model.py:143] round_filter input=16 output=24
I0628 08:21:17.288158 140347812091776 efficientnet_model.py:143] round_filter input=16 output=24
I0628 08:21:17.288279 140347812091776 efficientnet_model.py:143] round_filter input=24 output=32
I0628 08:21:17.515250 140347812091776 efficientnet_model.py:143] round_filter input=24 output=32
I0628 08:21:17.515388 140347812091776 efficientnet_model.py:143] round_filter input=40 output=48
I0628 08:21:17.731092 140347812091776 efficientnet_model.py:143] round_filter input=40 output=48
I0628 08:21:17.731232 140347812091776 efficientnet_model.py:143] round_filter input=80 output=96
I0628 08:21:18.118717 140347812091776 efficientnet_model.py:143] round_filter input=80 output=96
I0628 08:21:18.118894 140347812091776 efficientnet_model.py:143] round_filter input=112 output=136
I0628 08:21:18.695424 140347812091776 efficientnet_model.py:143] round_filter input=112 output=136
I0628 08:21:18.695630 140347812091776 efficientnet_model.py:143] round_filter input=192 output=232
I0628 08:21:19.196761 140347812091776 efficientnet_model.py:143] round_filter input=192 output=232
I0628 08:21:19.196946 140347812091776 efficientnet_model.py:143] round_filter input=320 output=384
I0628 08:21:19.349898 140347812091776 efficientnet_model.py:143] round_filter input=1280 output=1536
I0628 08:21:19.378389 140347812091776 efficientnet_model.py:453] Building model efficientnet with params ModelConfig(width_coefficient=1.2, depth_coefficient=1.4, resolution=300, dropout_rate=0.3, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
I0628 08:21:19.442692 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:146] EfficientDet EfficientNet backbone version: efficientnet-b4
I0628 08:21:19.442833 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 224
I0628 08:21:19.442907 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:149] EfficientDet BiFPN num iterations: 7
I0628 08:21:19.444375 140347812091776 efficientnet_model.py:143] round_filter input=32 output=48
I0628 08:21:19.459678 140347812091776 efficientnet_model.py:143] round_filter input=32 output=48
I0628 08:21:19.459788 140347812091776 efficientnet_model.py:143] round_filter input=16 output=24
I0628 08:21:19.580320 140347812091776 efficientnet_model.py:143] round_filter input=16 output=24
I0628 08:21:19.580437 140347812091776 efficientnet_model.py:143] round_filter input=24 output=32
I0628 08:21:19.887663 140347812091776 efficientnet_model.py:143] round_filter input=24 output=32
I0628 08:21:19.887821 140347812091776 efficientnet_model.py:143] round_filter input=40 output=56
I0628 08:21:20.194524 140347812091776 efficientnet_model.py:143] round_filter input=40 output=56
I0628 08:21:20.194688 140347812091776 efficientnet_model.py:143] round_filter input=80 output=112
I0628 08:21:20.650307 140347812091776 efficientnet_model.py:143] round_filter input=80 output=112
I0628 08:21:20.650466 140347812091776 efficientnet_model.py:143] round_filter input=112 output=160
I0628 08:21:21.101322 140347812091776 efficientnet_model.py:143] round_filter input=112 output=160
I0628 08:21:21.101495 140347812091776 efficientnet_model.py:143] round_filter input=192 output=272
I0628 08:21:21.706395 140347812091776 efficientnet_model.py:143] round_filter input=192 output=272
I0628 08:21:21.706574 140347812091776 efficientnet_model.py:143] round_filter input=320 output=448
I0628 08:21:21.853372 140347812091776 efficientnet_model.py:143] round_filter input=1280 output=1792
I0628 08:21:21.883697 140347812091776 efficientnet_model.py:453] Building model efficientnet with params ModelConfig(width_coefficient=1.4, depth_coefficient=1.8, resolution=380, dropout_rate=0.4, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
I0628 08:21:21.958340 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:146] EfficientDet EfficientNet backbone version: efficientnet-b5
I0628 08:21:21.958471 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 288
I0628 08:21:21.958538 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:149] EfficientDet BiFPN num iterations: 7
I0628 08:21:21.959994 140347812091776 efficientnet_model.py:143] round_filter input=32 output=48
I0628 08:21:21.974973 140347812091776 efficientnet_model.py:143] round_filter input=32 output=48
I0628 08:21:21.975070 140347812091776 efficientnet_model.py:143] round_filter input=16 output=24
I0628 08:21:22.155002 140347812091776 efficientnet_model.py:143] round_filter input=16 output=24
I0628 08:21:22.155180 140347812091776 efficientnet_model.py:143] round_filter input=24 output=40
I0628 08:21:22.534243 140347812091776 efficientnet_model.py:143] round_filter input=24 output=40
I0628 08:21:22.534408 140347812091776 efficientnet_model.py:143] round_filter input=40 output=64
I0628 08:21:22.916411 140347812091776 efficientnet_model.py:143] round_filter input=40 output=64
I0628 08:21:22.916581 140347812091776 efficientnet_model.py:143] round_filter input=80 output=128
I0628 08:21:23.454232 140347812091776 efficientnet_model.py:143] round_filter input=80 output=128
I0628 08:21:23.454401 140347812091776 efficientnet_model.py:143] round_filter input=112 output=176
I0628 08:21:24.187403 140347812091776 efficientnet_model.py:143] round_filter input=112 output=176
I0628 08:21:24.187578 140347812091776 efficientnet_model.py:143] round_filter input=192 output=304
I0628 08:21:24.891063 140347812091776 efficientnet_model.py:143] round_filter input=192 output=304
I0628 08:21:24.891237 140347812091776 efficientnet_model.py:143] round_filter input=320 output=512
I0628 08:21:25.113123 140347812091776 efficientnet_model.py:143] round_filter input=1280 output=2048
I0628 08:21:25.141521 140347812091776 efficientnet_model.py:453] Building model efficientnet with params ModelConfig(width_coefficient=1.6, depth_coefficient=2.2, resolution=456, dropout_rate=0.4, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
I0628 08:21:25.234969 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:146] EfficientDet EfficientNet backbone version: efficientnet-b6
I0628 08:21:25.235121 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 384
I0628 08:21:25.235191 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:149] EfficientDet BiFPN num iterations: 8
I0628 08:21:25.236783 140347812091776 efficientnet_model.py:143] round_filter input=32 output=56
I0628 08:21:25.253177 140347812091776 efficientnet_model.py:143] round_filter input=32 output=56
I0628 08:21:25.253283 140347812091776 efficientnet_model.py:143] round_filter input=16 output=32
I0628 08:21:25.434596 140347812091776 efficientnet_model.py:143] round_filter input=16 output=32
I0628 08:21:25.434759 140347812091776 efficientnet_model.py:143] round_filter input=24 output=40
I0628 08:21:25.892615 140347812091776 efficientnet_model.py:143] round_filter input=24 output=40
I0628 08:21:25.892822 140347812091776 efficientnet_model.py:143] round_filter input=40 output=72
I0628 08:21:26.362296 140347812091776 efficientnet_model.py:143] round_filter input=40 output=72
I0628 08:21:26.362469 140347812091776 efficientnet_model.py:143] round_filter input=80 output=144
I0628 08:21:26.962924 140347812091776 efficientnet_model.py:143] round_filter input=80 output=144
I0628 08:21:26.963091 140347812091776 efficientnet_model.py:143] round_filter input=112 output=200
I0628 08:21:27.582999 140347812091776 efficientnet_model.py:143] round_filter input=112 output=200
I0628 08:21:27.583170 140347812091776 efficientnet_model.py:143] round_filter input=192 output=344
I0628 08:21:28.431672 140347812091776 efficientnet_model.py:143] round_filter input=192 output=344
I0628 08:21:28.431839 140347812091776 efficientnet_model.py:143] round_filter input=320 output=576
I0628 08:21:28.646180 140347812091776 efficientnet_model.py:143] round_filter input=1280 output=2304
I0628 08:21:28.675453 140347812091776 efficientnet_model.py:453] Building model efficientnet with params ModelConfig(width_coefficient=1.8, depth_coefficient=2.6, resolution=528, dropout_rate=0.5, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
I0628 08:21:28.767486 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:146] EfficientDet EfficientNet backbone version: efficientnet-b7
I0628 08:21:28.767622 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:147] EfficientDet BiFPN num filters: 384
I0628 08:21:28.767692 140347812091776 ssd_efficientnet_bifpn_feature_extractor.py:149] EfficientDet BiFPN num iterations: 8
I0628 08:21:28.769284 140347812091776 efficientnet_model.py:143] round_filter input=32 output=64
I0628 08:21:28.784480 140347812091776 efficientnet_model.py:143] round_filter input=32 output=64
I0628 08:21:28.784585 140347812091776 efficientnet_model.py:143] round_filter input=16 output=32
I0628 08:21:29.030361 140347812091776 efficientnet_model.py:143] round_filter input=16 output=32
I0628 08:21:29.030514 140347812091776 efficientnet_model.py:143] round_filter input=24 output=48
I0628 08:21:29.789160 140347812091776 efficientnet_model.py:143] round_filter input=24 output=48
I0628 08:21:29.789331 140347812091776 efficientnet_model.py:143] round_filter input=40 output=80
I0628 08:21:30.329168 140347812091776 efficientnet_model.py:143] round_filter input=40 output=80
I0628 08:21:30.329329 140347812091776 efficientnet_model.py:143] round_filter input=80 output=160
I0628 08:21:31.091176 140347812091776 efficientnet_model.py:143] round_filter input=80 output=160
I0628 08:21:31.091342 140347812091776 efficientnet_model.py:143] round_filter input=112 output=224
I0628 08:21:31.847848 140347812091776 efficientnet_model.py:143] round_filter input=112 output=224
I0628 08:21:31.848006 140347812091776 efficientnet_model.py:143] round_filter input=192 output=384
I0628 08:21:32.832955 140347812091776 efficientnet_model.py:143] round_filter input=192 output=384
I0628 08:21:32.833122 140347812091776 efficientnet_model.py:143] round_filter input=320 output=640
I0628 08:21:33.139893 140347812091776 efficientnet_model.py:143] round_filter input=1280 output=2560
I0628 08:21:33.168404 140347812091776 efficientnet_model.py:453] Building model efficientnet with params ModelConfig(width_coefficient=2.0, depth_coefficient=3.1, resolution=600, dropout_rate=0.5, blocks=(BlockConfig(input_filters=32, output_filters=16, kernel_size=3, num_repeat=1, expand_ratio=1, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=16, output_filters=24, kernel_size=3, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=24, output_filters=40, kernel_size=5, num_repeat=2, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=40, output_filters=80, kernel_size=3, num_repeat=3, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=80, output_filters=112, kernel_size=5, num_repeat=3, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=112, output_filters=192, kernel_size=5, num_repeat=4, expand_ratio=6, strides=(2, 2), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise'), BlockConfig(input_filters=192, output_filters=320, kernel_size=3, num_repeat=1, expand_ratio=6, strides=(1, 1), se_ratio=0.25, id_skip=True, fused_conv=False, conv_type='depthwise')), stem_base_filters=32, top_base_filters=1280, activation='simple_swish', batch_norm='default', bn_momentum=0.99, bn_epsilon=0.001, weight_decay=5e-06, drop_connect_rate=0.2, depth_divisor=8, min_depth=None, use_se=True, input_channels=3, num_classes=1000, model_name='efficientnet', rescale_input=False, data_format='channels_last', dtype='float32')
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_create_ssd_models_from_config): 21.43s
I0628 08:21:33.276222 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_create_ssd_models_from_config): 21.43s
[       OK ] ModelBuilderTF2Test.test_create_ssd_models_from_config
[ RUN      ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update): 0.0s
I0628 08:21:33.281557 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update): 0.0s
[       OK ] ModelBuilderTF2Test.test_invalid_faster_rcnn_batchnorm_update
[ RUN      ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold): 0.0s
I0628 08:21:33.283310 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold): 0.0s
[       OK ] ModelBuilderTF2Test.test_invalid_first_stage_nms_iou_threshold
[ RUN      ] ModelBuilderTF2Test.test_invalid_model_config_proto
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_model_config_proto): 0.0s
I0628 08:21:33.283782 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_invalid_model_config_proto): 0.0s
[       OK ] ModelBuilderTF2Test.test_invalid_model_config_proto
[ RUN      ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_invalid_second_stage_batch_size): 0.0s
I0628 08:21:33.285165 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_invalid_second_stage_batch_size): 0.0s
[       OK ] ModelBuilderTF2Test.test_invalid_second_stage_batch_size
[ RUN      ] ModelBuilderTF2Test.test_session
[  SKIPPED ] ModelBuilderTF2Test.test_session
[ RUN      ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor): 0.0s
I0628 08:21:33.286439 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_faster_rcnn_feature_extractor
[ RUN      ] ModelBuilderTF2Test.test_unknown_meta_architecture
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
I0628 08:21:33.286894 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_unknown_meta_architecture): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_meta_architecture
[ RUN      ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
INFO:tensorflow:time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
I0628 08:21:33.287796 140347812091776 test_util.py:2459] time(__main__.ModelBuilderTF2Test.test_unknown_ssd_feature_extractor): 0.0s
[       OK ] ModelBuilderTF2Test.test_unknown_ssd_feature_extractor
----------------------------------------------------------------------
Ran 24 tests in 26.331s

OK (skipped=1)
Getting and processing the data
As mentioned before, the model is going to be trained using the Our own custom created dataset.

Run once if using google drive or once per session
!mkdir {path_root}/dataset
Extracting.zip images and annotations.zip
!unzip {path_root}"/dataset/images" -d {path_root}"/dataset"
Archive:  /content/drive/MyDrive/DSs/glasses_detection_project/MeGlass_Sorted_1024x1024_330_images.zip
   creating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (1).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (10).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (100).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (101).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (102).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (103).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (104).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (105).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (106).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (107).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (108).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (109).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (11).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (110).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (111).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (112).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (113).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (114).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (115).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (116).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (117).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (118).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (119).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (12).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (120).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (121).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (122).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (123).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (124).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (125).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (126).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (127).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (128).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (129).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (13).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (130).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (131).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (132).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (133).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (134).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (135).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (136).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (137).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (138).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (139).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (14).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (140).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (141).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (142).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (143).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (144).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (145).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (146).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (147).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (148).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (149).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (15).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (150).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (151).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (152).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (153).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (154).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (155).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (156).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (157).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (158).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (159).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (16).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (160).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (161).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (162).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (163).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (164).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (165).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (166).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (167).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (168).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (169).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (17).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (170).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (171).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (172).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (173).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (174).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (175).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (176).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (177).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (178).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (179).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (18).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (180).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (181).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (182).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (183).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (184).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (185).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (186).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (187).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (188).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (189).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (19).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (190).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (191).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (192).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (193).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (194).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (195).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (196).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (197).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (198).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (199).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (2).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (20).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (200).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (201).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (202).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (203).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (204).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (205).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (206).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (207).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (208).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (209).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (21).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (210).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (211).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (212).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (213).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (214).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (215).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (216).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (217).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (218).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (219).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (22).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (220).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (221).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (222).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (223).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (224).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (225).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (226).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (227).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (228).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (229).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (23).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (230).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (231).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (232).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (233).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (234).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (235).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (236).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (237).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (238).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (239).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (24).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (240).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (241).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (242).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (243).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (244).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (245).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (246).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (247).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (248).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (249).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (25).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (250).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (251).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (252).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (253).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (254).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (255).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (256).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (257).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (258).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (259).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (26).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (260).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (261).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (262).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (263).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (264).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (265).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (266).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (267).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (268).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (269).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (27).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (270).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (271).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (272).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (273).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (274).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (275).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (276).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (277).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (278).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (279).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (28).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (280).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (281).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (282).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (283).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (284).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (285).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (286).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (287).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (288).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (289).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (29).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (290).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (291).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (292).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (293).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (294).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (295).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (296).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (297).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (298).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (299).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (3).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (30).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (300).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (301).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (302).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (303).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (304).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (305).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (306).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (307).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (308).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (309).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (31).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (310).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (311).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (312).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (313).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (314).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (315).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (316).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (317).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (318).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (319).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (32).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (320).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (321).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (322).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (323).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (324).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (325).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (326).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (327).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (328).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (329).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (33).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (330).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (34).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (35).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (36).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (37).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (38).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (39).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (4).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (40).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (41).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (42).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (43).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (44).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (45).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (46).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (47).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (48).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (49).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (5).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (50).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (51).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (52).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (53).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (54).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (55).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (56).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (57).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (58).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (59).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (6).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (60).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (61).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (62).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (63).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (64).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (65).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (66).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (67).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (68).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (69).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (7).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (70).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (71).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (72).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (73).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (74).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (75).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (76).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (77).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (78).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (79).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (8).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (80).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (81).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (82).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (83).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (84).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (85).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (86).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (87).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (88).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (89).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (9).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (90).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (91).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (92).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (93).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (94).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (95).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (96).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (97).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (98).png  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_images/face (99).png  
!unzip {path_root}"/dataset/annotations" -d {path_root}"/dataset"
Archive:  /content/drive/MyDrive/DSs/glasses_detection_project/MeGlass_Sorted_1024x1024_330_annotations.zip
   creating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (1).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (10).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (100).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (101).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (102).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (103).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (104).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (105).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (106).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (107).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (108).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (109).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (11).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (110).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (111).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (112).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (113).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (114).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (115).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (116).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (117).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (118).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (119).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (12).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (120).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (121).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (122).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (123).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (124).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (125).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (126).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (127).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (128).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (129).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (13).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (130).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (131).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (132).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (133).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (134).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (135).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (136).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (137).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (138).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (139).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (14).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (140).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (141).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (142).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (143).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (144).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (145).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (146).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (147).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (148).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (149).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (15).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (150).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (151).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (152).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (153).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (154).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (155).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (156).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (157).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (158).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (159).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (16).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (160).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (161).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (162).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (163).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (164).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (165).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (166).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (167).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (168).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (169).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (17).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (170).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (171).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (172).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (173).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (174).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (175).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (176).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (177).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (178).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (179).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (18).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (180).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (181).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (182).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (183).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (184).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (185).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (186).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (187).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (188).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (189).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (19).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (190).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (191).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (192).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (193).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (194).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (195).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (196).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (197).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (198).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (199).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (2).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (20).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (200).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (201).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (202).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (203).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (204).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (205).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (206).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (207).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (208).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (209).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (21).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (210).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (211).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (212).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (213).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (214).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (215).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (216).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (217).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (218).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (219).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (22).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (220).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (221).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (222).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (223).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (224).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (225).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (226).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (227).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (228).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (229).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (23).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (230).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (231).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (232).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (233).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (234).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (235).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (236).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (237).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (238).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (239).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (24).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (240).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (241).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (242).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (243).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (244).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (245).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (246).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (247).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (248).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (249).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (25).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (250).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (251).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (252).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (253).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (254).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (255).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (256).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (257).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (258).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (259).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (26).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (260).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (261).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (262).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (263).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (264).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (265).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (266).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (267).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (268).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (269).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (27).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (270).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (271).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (272).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (273).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (274).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (275).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (276).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (277).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (278).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (279).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (28).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (280).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (281).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (282).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (283).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (284).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (285).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (286).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (287).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (288).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (289).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (29).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (290).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (291).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (292).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (293).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (294).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (295).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (296).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (297).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (298).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (299).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (3).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (30).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (300).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (301).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (302).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (303).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (304).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (305).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (306).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (307).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (308).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (309).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (31).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (310).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (311).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (312).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (313).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (314).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (315).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (316).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (317).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (318).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (319).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (32).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (320).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (321).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (322).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (323).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (324).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (325).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (326).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (327).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (328).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (329).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (33).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (330).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (34).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (35).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (36).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (37).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (38).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (39).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (4).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (40).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (41).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (42).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (43).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (44).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (45).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (46).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (47).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (48).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (49).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (5).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (50).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (51).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (52).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (53).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (54).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (55).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (56).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (57).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (58).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (59).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (6).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (60).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (61).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (62).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (63).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (64).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (65).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (66).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (67).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (68).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (69).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (7).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (70).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (71).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (72).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (73).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (74).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (75).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (76).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (77).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (78).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (79).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (8).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (80).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (81).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (82).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (83).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (84).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (85).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (86).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (87).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (88).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (89).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (9).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (90).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (91).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (92).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (93).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (94).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (95).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (96).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (97).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (98).xml  
  inflating: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/MeGlass_Sorted_1024x1024_330_annotations/face (99).xml  
Now, it’s necessary to create a labelmap file to define the classes that are going to be used. "glasses" is the only one, so right-click in the File section in dataset folder on Google Colab and create a New file named labelmap.pbtxt as follows:
item {
    name: "glasses"
    id: 1
}
The last step is to convert the data into a sequence of binary records so that they can be fed into Tensorflow’s object detection API. To do so, transform the data into the TFRecord format using the generate_tf_records.py script available in the Kangaroo Dataset:

Creating CSVs of XML labels:
Reference: https://gist.github.com/iKhushPatel/ed1f837656b155d9b94d45b42e00f5e4 - Cusomized as per our need
Run once if using google drive or once per session
import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df
  
xml_path = path_root + '/dataset/annotations'
df = xml_to_csv(xml_path)
df.head(3)
filename	width	height	class	xmin	ymin	xmax	ymax
0	face (1).png	1024	1024	glasses	196	437	810	578
1	face (10).png	1024	1024	glasses	225	444	788	552
2	face (100).png	1024	1024	glasses	215	432	837	601
df.shape
(330, 8)
Assigning out of 136 images we divide 5% for test and 95% for train
df_test = df.sample(frac = 0.10) # Assigning 33 out of 330 that is 10% images we divide 5% for test and 95% for train 
df_train = df[~df['filename'].isin(df_test['filename'])]
df_test.head(3)
filename	width	height	class	xmin	ymin	xmax	ymax
76	face (168).png	1024	1024	glasses	228	427	837	597
289	face (62).png	1024	1024	glasses	129	429	801	587
320	face (90).png	1024	1024	glasses	118	398	786	566
df_test.shape
(33, 8)
df_train.shape
(297, 8)
# Writing dataframe to csv
df_train.to_csv(path_root + '/dataset/train_labels.csv')
# Writing dataframe to csv
df_test.to_csv(path_root + '/dataset/test_labels.csv')
Creating record files of label CSVs:
Reference: https://gist.github.com/iKhushPatel/5614a36f26cf6459cc49c8248e8b5b48 - Cusomized as per our need -
Run once if using google drive or once per session
"""
Usage:
  # From tensorflow/models/
  # Create train data:
  python generate_tfrecord.py --csv_input=dataset/train_labels.csv  --output_path=dataset/train.record
  # Create test data:
  python generate_tfrecord.py --csv_input=dataset/test_labels.csv  --output_path=dataset/test.record
"""

from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import os
import io
import pandas as pd
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from PIL import Image
import sys
sys.path.append('../')
from object_detection.utils import dataset_util
from collections import namedtuple, OrderedDict


csv_input = path_root + '/dataset/train_labels.csv'
output_path = path_root + '/dataset/train.record'
image_dir = path_root + '/dataset/images/'


# TO-DO replace this with label map
def class_text_to_int(row_label):
    if row_label == 'Label1':
        return 1
    if row_label == 'Label2':
        return 2
    else:
        return 0


def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(group, path):
    with tf.gfile.GFile(os.path.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size

    filename = group.filename.encode('utf8')
    image_format = b'jpg'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for index, row in group.object.iterrows():
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(class_text_to_int(row['class']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example

# For train

writer = tf.python_io.TFRecordWriter(output_path)
path = os.path.join(image_dir)
examples = pd.read_csv(csv_input)
grouped = split(examples, 'filename')
for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path)
print('Successfully created the TFRecords: {}'.format(output_path))
Successfully created the TFRecords: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/train.record
# For test

csv_input = path_root + '/dataset/test_labels.csv'
output_path = path_root + '/dataset/test.record'

writer = tf.python_io.TFRecordWriter(output_path)
path = os.path.join(image_dir)
examples = pd.read_csv(csv_input)
grouped = split(examples, 'filename')
for group in grouped:
    tf_example = create_tf_example(group, path)
    writer.write(tf_example.SerializeToString())

writer.close()
output_path = os.path.join(os.getcwd(), output_path)
print('Successfully created the TFRecords: {}'.format(output_path))
Successfully created the TFRecords: /content/drive/MyDrive/DSs/glasses_detection_project/dataset/test.record
Choosing the model
We’re ready to choose the model that’s going to be the Kangaroo Detector. TensorFlow 2 provides 40 pre-trained detection models on the COCO 2017 Dataset. This collection is the TensorFlow 2 Detection Model Zoo and can be accessed here.

Every model has a Speed, Mean Average Precision(mAP) and Output. Generally, a higher mAP implies a lower speed, but as this project is based on a one-class object detection problem, the faster model (SSD MobileNet v2 320x320) should be enough.

Besides the Model Zoo, TensorFlow provides a Models Configs Repository as well. There, it’s possible to get the configuration file that has to be modified before the training. Let’s download the files:

%cd {path_root}/
/content/drive/.shortcut-targets-by-id/1ICDl8YeYQgBB5FK2XWH-e75qELCDjzqZ/DSs/glasses_detection_project
Run once if using google drive or once per session
!wget http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/mobilenet_v2.tar.gz
!tar -xvf mobilenet_v2.tar.gz
!rm mobilenet_v2.tar.gz
--2022-06-27 11:02:30--  http://download.tensorflow.org/models/object_detection/classification/tf2/20200710/mobilenet_v2.tar.gz
Resolving download.tensorflow.org (download.tensorflow.org)... 142.250.188.48, 2607:f8b0:4004:800::2010
Connecting to download.tensorflow.org (download.tensorflow.org)|142.250.188.48|:80... connected.
HTTP request sent, awaiting response... 200 OK
Length: 8404070 (8.0M) [application/x-tar]
Saving to: ‘mobilenet_v2.tar.gz’

mobilenet_v2.tar.gz 100%[===================>]   8.01M  48.4MB/s    in 0.2s    

2022-06-27 11:02:31 (48.4 MB/s) - ‘mobilenet_v2.tar.gz’ saved [8404070/8404070]

mobilenet_v2/
mobilenet_v2/mobilenet_v2.ckpt-1.index
mobilenet_v2/checkpoint
mobilenet_v2/mobilenet_v2.ckpt-1.data-00001-of-00002
mobilenet_v2/mobilenet_v2.ckpt-1.data-00000-of-00002
!wget https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config
!mv ssd_mobilenet_v2_320x320_coco17_tpu-8.config mobilenet_v2.config
--2022-06-27 11:02:35--  https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/configs/tf2/ssd_mobilenet_v2_320x320_coco17_tpu-8.config
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 4484 (4.4K) [text/plain]
Saving to: ‘ssd_mobilenet_v2_320x320_coco17_tpu-8.config’

ssd_mobilenet_v2_32 100%[===================>]   4.38K  --.-KB/s    in 0.001s  

2022-06-27 11:02:35 (4.96 MB/s) - ‘ssd_mobilenet_v2_320x320_coco17_tpu-8.config’ saved [4484/4484]

Configure training - Run once per session
As mentioned before, the downloaded weights were pre-trained on the COCO 2017 Dataset, but the focus here is to train the model to recognize one class so these weights are going to be used only to initialize the network — this technique is known as transfer learning, and it’s commonly used to speed up the learning process.

From now, what has to be done is to set up the mobilenet_v2.config file, and start the training. I highly recommend reading the MobileNetV2 paper (Sandler, Mark, et al. - 2018) to get the gist of the architecture.

Choosing the best hyperparameters is a task that requires some experimentation. As the resources are limited in the Google Colab, I am going to use the same batch size as the paper, set a number of steps to get a reasonably low loss, and leave all the other values as default. If you want to try something more sophisticated to find the hyperparameters, I recommend Keras Tuner - an easy-to-use framework that applies Bayesian Optimization, Hyperband, and Random Search algorithms.

Defining training parameters
num_classes = 1
batch_size = 60
num_steps = 15000
num_eval_steps = 1000

train_record_path = path_root+'/dataset/train.record'
test_record_path = path_root+'/dataset/test.record'
model_dir = path_root+'/training/'
labelmap_path = path_root+'/dataset/labelmap.pbtxt'

pipeline_config_path = path_root+'/mobilenet_v2.config'
fine_tune_checkpoint = path_root+'/mobilenet_v2/mobilenet_v2.ckpt-1'
Editing config file
import re

with open(pipeline_config_path) as f:
    config = f.read()

with open(pipeline_config_path, 'w') as f:

  # Set labelmap path
  config = re.sub('label_map_path: ".*?"', 
             'label_map_path: "{}"'.format(labelmap_path), config)
  
  # Set fine_tune_checkpoint path
  config = re.sub('fine_tune_checkpoint: ".*?"',
                  'fine_tune_checkpoint: "{}"'.format(fine_tune_checkpoint), config)
  
  # Set train tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/train)(.*?")', 
                  'input_path: "{}"'.format(train_record_path), config)
  
  # Set test tf-record file path
  config = re.sub('(input_path: ".*?)(PATH_TO_BE_CONFIGURED/val)(.*?")', 
                  'input_path: "{}"'.format(test_record_path), config)
  
  # Set number of classes.
  config = re.sub('num_classes: [0-9]+',
                  'num_classes: {}'.format(num_classes), config)
  
  # Set batch size
  config = re.sub('batch_size: [0-9]+',
                  'batch_size: {}'.format(batch_size), config)
  
  # Set training steps
  config = re.sub('num_steps: [0-9]+',
                  'num_steps: {}'.format(num_steps), config)
  
  f.write(config)
Error Solving:
"ImportError: cannot import name '_registerMatType' from 'cv2.cv2' (/usr/local/lib/python3.7/dist-packages/cv2/cv2.cpython-37m-x86_64-linux-gnu.so)"

!pip uninstall opencv-python-headless==4.5.5.62
Found existing installation: opencv-python-headless 4.5.2.52
Uninstalling opencv-python-headless-4.5.2.52:
  Would remove:
    /usr/local/lib/python3.7/dist-packages/cv2/*
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless-4.5.2.52.dist-info/*
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libavcodec-8daa01ff.so.58.109.100
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libavformat-06a336f2.so.58.61.100
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libavutil-01d48d95.so.56.60.100
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libbz2-a273e504.so.1.0.6
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libcrypto-098682aa.so.1.1
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libpng15-c2ffaf3d.so.15.13.0
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libssl-f3db6a3b.so.1.1
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libswresample-4767dc06.so.3.8.100
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libswscale-2d2bce5d.so.5.8.100
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libvpx-14094576.so.6.3.0
    /usr/local/lib/python3.7/dist-packages/opencv_python_headless.libs/libz-d8a329de.so.1.2.7
  Would not remove (might be manually added):
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libQtCore-bbdab771.so.4.8.7
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libQtGui-903938cd.so.4.8.7
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libQtTest-1183da5d.so.4.8.7
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libavcodec-3cdd3bd4.so.58.62.100
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libavformat-69a63b50.so.58.35.100
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libavutil-8e8979a8.so.56.36.100
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libbz2-7225278b.so.1.0.3
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libcrypto-a25ff511.so.1.1
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libssl-fdf0b66c.so.1.1
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libswresample-c6b3bbb9.so.3.6.100
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libswscale-2d19f7d1.so.5.6.100
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libvpx-c887ea55.so.6.1.0
    /usr/local/lib/python3.7/dist-packages/cv2/.libs/libz-a147dcb0.so.1.2.3
Proceed (y/n)? y
  Successfully uninstalled opencv-python-headless-4.5.2.52
!pip install opencv-python-headless==4.5.2.52
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting opencv-python-headless==4.5.2.52
  Using cached opencv_python_headless-4.5.2.52-cp37-cp37m-manylinux2014_x86_64.whl (38.2 MB)
Requirement already satisfied: numpy>=1.14.5 in /usr/local/lib/python3.7/dist-packages (from opencv-python-headless==4.5.2.52) (1.21.6)
Installing collected packages: opencv-python-headless
Successfully installed opencv-python-headless-4.5.2.52
Error solving:
"Node: 'ssd_mobile_net_v2_keras_feature_extractor/model/Conv1/Conv2D' DNN library is not found."

!pip install tensorflow==2.7.0
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Collecting tensorflow==2.7.0
  Downloading https://us-python.pkg.dev/colab-wheels/public/tensorflow/tensorflow-2.7.0%2Bzzzcolab20220506150900-cp37-cp37m-linux_x86_64.whl
     | 665.5 MB 101.3 MB/s
Requirement already satisfied: six>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.15.0)
Requirement already satisfied: flatbuffers<3.0,>=1.12 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.12)
Collecting tensorflow-estimator<2.8,~=2.7.0rc0
  Using cached tensorflow_estimator-2.7.0-py2.py3-none-any.whl (463 kB)
Requirement already satisfied: tensorboard~=2.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (2.9.1)
Requirement already satisfied: protobuf>=3.9.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (3.19.4)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.6.3)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.1.0)
Requirement already satisfied: numpy>=1.14.5 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.21.6)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.46.3)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (0.26.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (3.3.0)
Requirement already satisfied: absl-py>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.1.0)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.14.1)
Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (1.1.2)
Requirement already satisfied: gast<0.5.0,>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (0.4.0)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (4.1.1)
Requirement already satisfied: wheel<1.0,>=0.32.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (0.37.1)
Requirement already satisfied: libclang>=9.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (14.0.1)
Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (3.1.0)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow==2.7.0) (0.2.0)
Collecting keras<2.8,>=2.7.0rc0
  Using cached keras-2.7.0-py2.py3-none-any.whl (1.3 MB)
Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow==2.7.0) (1.5.2)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (0.6.1)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (1.8.1)
Requirement already satisfied: setuptools>=41.0.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (57.4.0)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (0.4.6)
Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (1.35.0)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (2.28.0)
Requirement already satisfied: werkzeug>=1.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (1.0.1)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard~=2.6->tensorflow==2.7.0) (3.3.7)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.0) (4.2.4)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.0) (0.2.8)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.0) (4.8)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow==2.7.0) (1.3.1)
Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard~=2.6->tensorflow==2.7.0) (4.11.4)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard~=2.6->tensorflow==2.7.0) (3.8.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard~=2.6->tensorflow==2.7.0) (0.4.8)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.0) (2022.6.15)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.0) (2.10)
Requirement already satisfied: charset-normalizer~=2.0.0 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.0) (2.0.12)
Requirement already satisfied: urllib3<1.27,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard~=2.6->tensorflow==2.7.0) (1.24.3)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard~=2.6->tensorflow==2.7.0) (3.2.0)
Installing collected packages: tensorflow-estimator, keras, tensorflow
  Attempting uninstall: tensorflow-estimator
    Found existing installation: tensorflow-estimator 2.9.0
    Uninstalling tensorflow-estimator-2.9.0:
      Successfully uninstalled tensorflow-estimator-2.9.0
  Attempting uninstall: keras
    Found existing installation: keras 2.9.0
    Uninstalling keras-2.9.0:
      Successfully uninstalled keras-2.9.0
  Attempting uninstall: tensorflow
    Found existing installation: tensorflow 2.9.1
    Uninstalling tensorflow-2.9.1:
      Successfully uninstalled tensorflow-2.9.1
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
tf-models-official 2.9.2 requires tensorflow~=2.9.0, but you have tensorflow 2.7.0+zzzcolab20220506150900 which is incompatible.
tensorflow-text 2.9.0 requires tensorflow<2.10,>=2.9.0; platform_machine != "arm64" or platform_system != "Darwin", but you have tensorflow 2.7.0+zzzcolab20220506150900 which is incompatible.
Successfully installed keras-2.7.0 tensorflow-2.7.0+zzzcolab20220506150900 tensorflow-estimator-2.7.0
Caution: Below code block Removes the training folder and all its contents - Removing the train folder - Optional if you want to retrain.
#!rm -r {path_root}/training
With the parameters set, start the training:
!cd {path_root}
!python {path_root}/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_config_path} \
    --model_dir={model_dir} \
    --alsologtostderr \
    --num_train_steps={num_steps} \
    --sample_1_of_n_eval_examples=1 \
    --num_eval_steps={num_eval_steps}
2022-06-28 12:58:28.018720: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
INFO:tensorflow:Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
I0628 12:58:28.036121 140087626643328 mirrored_strategy.py:376] Using MirroredStrategy with devices ('/job:localhost/replica:0/task:0/device:GPU:0',)
INFO:tensorflow:Maybe overwriting train_steps: 15000
I0628 12:58:28.044157 140087626643328 config_util.py:552] Maybe overwriting train_steps: 15000
INFO:tensorflow:Maybe overwriting use_bfloat16: False
I0628 12:58:28.044348 140087626643328 config_util.py:552] Maybe overwriting use_bfloat16: False
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/object_detection/model_lib_v2.py:564: StrategyBase.experimental_distribute_datasets_from_function (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
Instructions for updating:
rename to distribute_datasets_from_function
W0628 12:58:28.406404 140087626643328 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/object_detection/model_lib_v2.py:564: StrategyBase.experimental_distribute_datasets_from_function (from tensorflow.python.distribute.distribute_lib) is deprecated and will be removed in a future version.
Instructions for updating:
rename to distribute_datasets_from_function
INFO:tensorflow:Reading unweighted datasets: ['/content/drive/MyDrive/DSs/glasses_detection_project/dataset/train.record']
I0628 12:58:28.726060 140087626643328 dataset_builder.py:162] Reading unweighted datasets: ['/content/drive/MyDrive/DSs/glasses_detection_project/dataset/train.record']
INFO:tensorflow:Reading record datasets for input file: ['/content/drive/MyDrive/DSs/glasses_detection_project/dataset/train.record']
I0628 12:58:28.726695 140087626643328 dataset_builder.py:79] Reading record datasets for input file: ['/content/drive/MyDrive/DSs/glasses_detection_project/dataset/train.record']
INFO:tensorflow:Number of filenames to read: 1
I0628 12:58:28.726831 140087626643328 dataset_builder.py:80] Number of filenames to read: 1
WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
W0628 12:58:28.726891 140087626643328 dataset_builder.py:87] num_readers has been reduced to 1 to match input file shards.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/object_detection/builders/dataset_builder.py:104: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.
W0628 12:58:28.730661 140087626643328 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/object_detection/builders/dataset_builder.py:104: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/object_detection/builders/dataset_builder.py:236: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.map()
W0628 12:58:28.836769 140087626643328 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/object_detection/builders/dataset_builder.py:236: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.map()
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1096: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
W0628 12:58:41.032697 140087626643328 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1096: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1096: sample_distorted_bounding_box (from tensorflow.python.ops.image_ops_impl) is deprecated and will be removed in a future version.
Instructions for updating:
`seed2` arg is deprecated.Use sample_distorted_bounding_box_v2 instead.
W0628 12:58:44.296398 140087626643328 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1096: sample_distorted_bounding_box (from tensorflow.python.ops.image_ops_impl) is deprecated and will be removed in a future version.
Instructions for updating:
`seed2` arg is deprecated.Use sample_distorted_bounding_box_v2 instead.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py:465: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
W0628 12:58:46.975518 140087626643328 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py:465: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
2022-06-28 12:59:13.676281: W tensorflow/core/framework/op_kernel.cc:1745] OP_REQUIRES failed at multi_device_iterator_ops.cc:789 : NOT_FOUND: Resource AnonymousMultiDeviceIterator/AnonymousMultiDeviceIterator0/N10tensorflow4data12_GLOBAL__N_119MultiDeviceIteratorE does not exist.
/usr/local/lib/python3.7/dist-packages/keras/backend.py:414: UserWarning: `tf.keras.backend.set_learning_phase` is deprecated and will be removed after 2020-10-11. To update it, simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
  warnings.warn('`tf.keras.backend.set_learning_phase` is deprecated and '
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 12:59:27.068189 140082550355712 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 12:59:27.068540 140082550355712 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 12:59:27.068713 140082550355712 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 12:59:27.068857 140082550355712 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 12:59:27.068992 140082550355712 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 12:59:27.069123 140082550355712 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/deprecation.py:620: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Use fn_output_signature instead
W0628 13:00:22.062795 140082474821376 deprecation.py:551] From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/deprecation.py:620: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with dtype is deprecated and will be removed in a future version.
Instructions for updating:
Use fn_output_signature instead
INFO:tensorflow:Step 14100 per-step time 3.287s
I0628 13:05:50.282229 140087626643328 model_lib_v2.py:707] Step 14100 per-step time 3.287s
INFO:tensorflow:{'Loss/classification_loss': 0.056255978,
 'Loss/localization_loss': 0.01783674,
 'Loss/regularization_loss': 0.046961743,
 'Loss/total_loss': 0.12105446,
 'learning_rate': 0.68098545}
I0628 13:05:50.288337 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.056255978,
 'Loss/localization_loss': 0.01783674,
 'Loss/regularization_loss': 0.046961743,
 'Loss/total_loss': 0.12105446,
 'learning_rate': 0.68098545}
INFO:tensorflow:Step 14200 per-step time 2.220s
I0628 13:09:32.315052 140087626643328 model_lib_v2.py:707] Step 14200 per-step time 2.220s
INFO:tensorflow:{'Loss/classification_loss': 0.047840063,
 'Loss/localization_loss': 0.017953908,
 'Loss/regularization_loss': 0.04685058,
 'Loss/total_loss': 0.11264455,
 'learning_rate': 0.6791162}
I0628 13:09:32.315546 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.047840063,
 'Loss/localization_loss': 0.017953908,
 'Loss/regularization_loss': 0.04685058,
 'Loss/total_loss': 0.11264455,
 'learning_rate': 0.6791162}
INFO:tensorflow:Step 14300 per-step time 2.221s
I0628 13:13:14.398948 140087626643328 model_lib_v2.py:707] Step 14300 per-step time 2.221s
INFO:tensorflow:{'Loss/classification_loss': 0.053745594,
 'Loss/localization_loss': 0.015343509,
 'Loss/regularization_loss': 0.046323646,
 'Loss/total_loss': 0.11541275,
 'learning_rate': 0.67723495}
I0628 13:13:14.399368 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.053745594,
 'Loss/localization_loss': 0.015343509,
 'Loss/regularization_loss': 0.046323646,
 'Loss/total_loss': 0.11541275,
 'learning_rate': 0.67723495}
INFO:tensorflow:Step 14400 per-step time 2.273s
I0628 13:17:01.749733 140087626643328 model_lib_v2.py:707] Step 14400 per-step time 2.273s
INFO:tensorflow:{'Loss/classification_loss': 0.06100835,
 'Loss/localization_loss': 0.018398853,
 'Loss/regularization_loss': 0.045790523,
 'Loss/total_loss': 0.12519772,
 'learning_rate': 0.6753418}
I0628 13:17:01.750156 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.06100835,
 'Loss/localization_loss': 0.018398853,
 'Loss/regularization_loss': 0.045790523,
 'Loss/total_loss': 0.12519772,
 'learning_rate': 0.6753418}
INFO:tensorflow:Step 14500 per-step time 2.152s
I0628 13:20:36.981640 140087626643328 model_lib_v2.py:707] Step 14500 per-step time 2.152s
INFO:tensorflow:{'Loss/classification_loss': 0.060440537,
 'Loss/localization_loss': 0.01952271,
 'Loss/regularization_loss': 0.04521458,
 'Loss/total_loss': 0.12517783,
 'learning_rate': 0.67343694}
I0628 13:20:36.982041 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.060440537,
 'Loss/localization_loss': 0.01952271,
 'Loss/regularization_loss': 0.04521458,
 'Loss/total_loss': 0.12517783,
 'learning_rate': 0.67343694}
INFO:tensorflow:Step 14600 per-step time 2.339s
I0628 13:24:30.839119 140087626643328 model_lib_v2.py:707] Step 14600 per-step time 2.339s
INFO:tensorflow:{'Loss/classification_loss': 0.059858736,
 'Loss/localization_loss': 0.022943638,
 'Loss/regularization_loss': 0.044753563,
 'Loss/total_loss': 0.12755594,
 'learning_rate': 0.6715203}
I0628 13:24:30.839528 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.059858736,
 'Loss/localization_loss': 0.022943638,
 'Loss/regularization_loss': 0.044753563,
 'Loss/total_loss': 0.12755594,
 'learning_rate': 0.6715203}
INFO:tensorflow:Step 14700 per-step time 2.218s
I0628 13:28:12.669953 140087626643328 model_lib_v2.py:707] Step 14700 per-step time 2.218s
INFO:tensorflow:{'Loss/classification_loss': 0.047191106,
 'Loss/localization_loss': 0.012824946,
 'Loss/regularization_loss': 0.04444419,
 'Loss/total_loss': 0.10446024,
 'learning_rate': 0.6695921}
I0628 13:28:12.670378 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.047191106,
 'Loss/localization_loss': 0.012824946,
 'Loss/regularization_loss': 0.04444419,
 'Loss/total_loss': 0.10446024,
 'learning_rate': 0.6695921}
INFO:tensorflow:Step 14800 per-step time 2.242s
I0628 13:31:56.897108 140087626643328 model_lib_v2.py:707] Step 14800 per-step time 2.242s
INFO:tensorflow:{'Loss/classification_loss': 0.0611878,
 'Loss/localization_loss': 0.012371646,
 'Loss/regularization_loss': 0.04398424,
 'Loss/total_loss': 0.11754368,
 'learning_rate': 0.66765225}
I0628 13:31:56.897581 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.0611878,
 'Loss/localization_loss': 0.012371646,
 'Loss/regularization_loss': 0.04398424,
 'Loss/total_loss': 0.11754368,
 'learning_rate': 0.66765225}
INFO:tensorflow:Step 14900 per-step time 2.240s
I0628 13:35:40.853831 140087626643328 model_lib_v2.py:707] Step 14900 per-step time 2.240s
INFO:tensorflow:{'Loss/classification_loss': 0.059303977,
 'Loss/localization_loss': 0.018075055,
 'Loss/regularization_loss': 0.04340902,
 'Loss/total_loss': 0.12078805,
 'learning_rate': 0.665701}
I0628 13:35:40.854247 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.059303977,
 'Loss/localization_loss': 0.018075055,
 'Loss/regularization_loss': 0.04340902,
 'Loss/total_loss': 0.12078805,
 'learning_rate': 0.665701}
INFO:tensorflow:Step 15000 per-step time 2.217s
I0628 13:39:22.559793 140087626643328 model_lib_v2.py:707] Step 15000 per-step time 2.217s
INFO:tensorflow:{'Loss/classification_loss': 0.053538267,
 'Loss/localization_loss': 0.020464603,
 'Loss/regularization_loss': 0.04300645,
 'Loss/total_loss': 0.11700931,
 'learning_rate': 0.6637383}
I0628 13:39:22.560184 140087626643328 model_lib_v2.py:708] {'Loss/classification_loss': 0.053538267,
 'Loss/localization_loss': 0.020464603,
 'Loss/regularization_loss': 0.04300645,
 'Loss/total_loss': 0.11700931,
 'learning_rate': 0.6637383}
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.000189 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.014858 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.017538 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.018674 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.028124 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.029279 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.032215 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.033324 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.038129 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
INFO:tensorflow:Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
I0628 13:39:23.055257 140087626643328 cross_device_ops.py:621] Reduce to /job:localhost/replica:0/task:0/device:CPU:0 then broadcast to ('/job:localhost/replica:0/task:0/device:CPU:0',).
To identify how well the training is going, we use the loss value. Loss is a number indicating how bad the model’s prediction was on the training samples. If the model’s prediction is perfect, the loss is zero; otherwise, the loss is greater. The goal of training a model is to find a set of weights and biases that have low loss, on average, across all examples (Descending into ML: Training and Loss | Machine Learning Crash Course).

From the logs, it’s possible to see a downward trend in the values so we say that “The model is converging”. In the next section, we’re going to plot these values for all training steps and the trend will be even clearer.

The model took around 4h to train (with Colab GPU), but by setting different parameters, you can make the process faster or slower. Everything depends on the number of classes you are using and your Precision/Recall target. A highly accurate network that recognizes multiple classes will take more steps and require more detailed parameters tuning.

Validate the model
Now let’s evaluate the trained model using the test data:

Here we're going to run the code through a loop that waits for checkpoints to evaluate. Once the evaluation finishes, you're going to see the message:
INFO:tensorflow:Waiting for new checkpoint at /< your_project_folder >/training/

Then you can stop the cell

!python {path_root}/models/research/object_detection/model_main_tf2.py \
    --pipeline_config_path={pipeline_config_path} \
    --model_dir={model_dir} \
    --checkpoint_dir={model_dir}
WARNING:tensorflow:Forced number of epochs for all eval validations to be 1.
W0628 13:51:26.253338 140318393227136 model_lib_v2.py:1090] Forced number of epochs for all eval validations to be 1.
INFO:tensorflow:Maybe overwriting sample_1_of_n_eval_examples: None
I0628 13:51:26.253615 140318393227136 config_util.py:552] Maybe overwriting sample_1_of_n_eval_examples: None
INFO:tensorflow:Maybe overwriting use_bfloat16: False
I0628 13:51:26.253706 140318393227136 config_util.py:552] Maybe overwriting use_bfloat16: False
INFO:tensorflow:Maybe overwriting eval_num_epochs: 1
I0628 13:51:26.253788 140318393227136 config_util.py:552] Maybe overwriting eval_num_epochs: 1
WARNING:tensorflow:Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
W0628 13:51:26.253911 140318393227136 model_lib_v2.py:1110] Expected number of evaluation epochs is 1, but instead encountered `eval_on_train_input_config.num_epochs` = 0. Overwriting `num_epochs` to 1.
2022-06-28 13:51:29.565875: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
INFO:tensorflow:Reading unweighted datasets: ['/content/drive/MyDrive/DSs/glasses_detection_project/dataset/test.record']
I0628 13:51:29.748188 140318393227136 dataset_builder.py:162] Reading unweighted datasets: ['/content/drive/MyDrive/DSs/glasses_detection_project/dataset/test.record']
INFO:tensorflow:Reading record datasets for input file: ['/content/drive/MyDrive/DSs/glasses_detection_project/dataset/test.record']
I0628 13:51:29.748830 140318393227136 dataset_builder.py:79] Reading record datasets for input file: ['/content/drive/MyDrive/DSs/glasses_detection_project/dataset/test.record']
INFO:tensorflow:Number of filenames to read: 1
I0628 13:51:29.748980 140318393227136 dataset_builder.py:80] Number of filenames to read: 1
WARNING:tensorflow:num_readers has been reduced to 1 to match input file shards.
W0628 13:51:29.749050 140318393227136 dataset_builder.py:87] num_readers has been reduced to 1 to match input file shards.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/object_detection/builders/dataset_builder.py:104: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.
W0628 13:51:29.750956 140318393227136 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/object_detection/builders/dataset_builder.py:104: parallel_interleave (from tensorflow.python.data.experimental.ops.interleave_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.interleave(map_func, cycle_length, block_length, num_parallel_calls=tf.data.AUTOTUNE)` instead. If sloppy execution is desired, use `tf.data.Options.deterministic`.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/object_detection/builders/dataset_builder.py:236: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.map()
W0628 13:51:29.782676 140318393227136 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/object_detection/builders/dataset_builder.py:236: DatasetV1.map_with_legacy_function (from tensorflow.python.data.ops.dataset_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.data.Dataset.map()
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1096: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
W0628 13:51:33.978860 140318393227136 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py:1096: sparse_to_dense (from tensorflow.python.ops.sparse_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Create a `tf.sparse.SparseTensor` and use `tf.sparse.to_dense` instead.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py:465: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
W0628 13:51:35.089432 140318393227136 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py:465: to_float (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
INFO:tensorflow:Waiting for new checkpoint at /content/drive/MyDrive/DSs/glasses_detection_project/training/
I0628 13:51:37.557292 140318393227136 checkpoint_utils.py:140] Waiting for new checkpoint at /content/drive/MyDrive/DSs/glasses_detection_project/training/
INFO:tensorflow:Found new checkpoint at /content/drive/MyDrive/DSs/glasses_detection_project/training/ckpt-19
I0628 13:51:37.561070 140318393227136 checkpoint_utils.py:149] Found new checkpoint at /content/drive/MyDrive/DSs/glasses_detection_project/training/ckpt-19
/usr/local/lib/python3.7/dist-packages/keras/backend.py:414: UserWarning: `tf.keras.backend.set_learning_phase` is deprecated and will be removed after 2020-10-11. To update it, simply pass a True/False value to the `training` argument of the `__call__` method of your layer or model.
  warnings.warn('`tf.keras.backend.set_learning_phase` is deprecated and '
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:51:45.696900 140318393227136 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:51:45.697385 140318393227136 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:51:45.697706 140318393227136 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:51:45.697961 140318393227136 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:51:45.698184 140318393227136 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:51:45.698408 140318393227136 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/object_detection/eval_util.py:929: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
W0628 13:52:07.726733 140318393227136 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/object_detection/eval_util.py:929: to_int64 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
Instructions for updating:
Use `tf.cast` instead.
INFO:tensorflow:Finished eval step 0
I0628 13:52:07.736448 140318393227136 model_lib_v2.py:966] Finished eval step 0
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py:465: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    
W0628 13:52:07.891050 140318393227136 deprecation.py:347] From /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py:465: py_func (from tensorflow.python.ops.script_ops) is deprecated and will be removed in a future version.
Instructions for updating:
tf.py_func is deprecated in TF V2. Instead, there are two
    options available in V2.
    - tf.py_function takes a python function which manipulates tf eager
    tensors instead of numpy arrays. It's easy to convert a tf eager tensor to
    an ndarray (just call tensor.numpy()) but having access to eager tensors
    means `tf.py_function`s can use accelerators such as GPUs as well as
    being differentiable using a gradient tape.
    - tf.numpy_function maintains the semantics of the deprecated tf.py_func
    (it is not differentiable, and manipulates numpy arrays). It drops the
    stateful argument making all functions stateful.
    
INFO:tensorflow:Performing evaluation on 33 images.
I0628 13:52:18.890955 140318393227136 coco_evaluation.py:293] Performing evaluation on 33 images.
creating index...
index created!
INFO:tensorflow:Loading and preparing annotation results...
I0628 13:52:18.891480 140318393227136 coco_tools.py:116] Loading and preparing annotation results...
INFO:tensorflow:DONE (t=0.00s)
I0628 13:52:18.893446 140318393227136 coco_tools.py:138] DONE (t=0.00s)
creating index...
index created!
Running per image evaluation...
Evaluate annotation type *bbox*
DONE (t=0.13s).
Accumulating evaluation results...
DONE (t=0.02s).
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.865
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.865
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.888
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.888
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.888
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.888
INFO:tensorflow:Eval metrics at step 15000
I0628 13:52:19.050863 140318393227136 model_lib_v2.py:1015] Eval metrics at step 15000
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP: 0.865352
I0628 13:52:19.062465 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Precision/mAP: 0.865352
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP@.50IOU: 1.000000
I0628 13:52:19.064033 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Precision/mAP@.50IOU: 1.000000
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP@.75IOU: 1.000000
I0628 13:52:19.065219 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Precision/mAP@.75IOU: 1.000000
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (small): -1.000000
I0628 13:52:19.066486 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Precision/mAP (small): -1.000000
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (medium): -1.000000
I0628 13:52:19.067580 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Precision/mAP (medium): -1.000000
INFO:tensorflow:	+ DetectionBoxes_Precision/mAP (large): 0.865352
I0628 13:52:19.068664 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Precision/mAP (large): 0.865352
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@1: 0.887879
I0628 13:52:19.069927 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Recall/AR@1: 0.887879
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@10: 0.887879
I0628 13:52:19.071058 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Recall/AR@10: 0.887879
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100: 0.887879
I0628 13:52:19.072147 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Recall/AR@100: 0.887879
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (small): -1.000000
I0628 13:52:19.073222 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Recall/AR@100 (small): -1.000000
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (medium): -1.000000
I0628 13:52:19.074298 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Recall/AR@100 (medium): -1.000000
INFO:tensorflow:	+ DetectionBoxes_Recall/AR@100 (large): 0.887879
I0628 13:52:19.075541 140318393227136 model_lib_v2.py:1018] 	+ DetectionBoxes_Recall/AR@100 (large): 0.887879
INFO:tensorflow:	+ Loss/localization_loss: 0.017211
I0628 13:52:19.076538 140318393227136 model_lib_v2.py:1018] 	+ Loss/localization_loss: 0.017211
INFO:tensorflow:	+ Loss/classification_loss: 0.109846
I0628 13:52:19.077554 140318393227136 model_lib_v2.py:1018] 	+ Loss/classification_loss: 0.109846
INFO:tensorflow:	+ Loss/regularization_loss: 0.043006
I0628 13:52:19.078571 140318393227136 model_lib_v2.py:1018] 	+ Loss/regularization_loss: 0.043006
INFO:tensorflow:	+ Loss/total_loss: 0.170063
I0628 13:52:19.079576 140318393227136 model_lib_v2.py:1018] 	+ Loss/total_loss: 0.170063
INFO:tensorflow:Exiting evaluation at step 15000
I0628 13:52:19.209238 140318393227136 model_lib_v2.py:1168] Exiting evaluation at step 15000
The evaluation was done in 7 images and provides three metrics based on the COCO detection evaluation metrics: Precision, Recall and Loss.
The Recall measures how good the model is at hitting the positive class, That is, from the positive samples, how many did the algorithm get right?

Recall

Precision defines how much you can rely on the positive class prediction: From the samples that the model said were positive, how many actually are?

Precision

Setting a practical example: Imagine we have an image containing 10 kangaroos, our model returned 5 detections, being 3 real kangaroos (TP = 3, FN =7) and 2 wrong detections (FP = 2). In that case, we have a 30% recall (the model detected 3 out of 10 kangaroos in the image) and a 60% precision (from the 5 detections, 3 were correct).

The precision and recall were divided by Intersection over Union (IoU) thresholds. The IoU is defined as the area of the intersection divided by the area of the union of a predicted bounding box (B) to a ground-truth box (B)(Zeng, N. - 2018):

Intersection over Union

For simplicity, it’s possible to consider that the IoU thresholds are used to determine whether a detection is a true positive(TP), a false positive(FP) or a false negative (FN). See an example below:

IoU threshold examples

With these concepts in mind, we can analyze some of the metrics we got from the evaluation. From the TensorFlow 2 Detection Model Zoo, the SSD MobileNet v2 320x320 has an mAP of 0.202. Our model presented the following average precisions (AP) for different IoUs:

AP@[IoU=0.50:0.95 | area=all | maxDets=100] = 0.222 AP@[IoU=0.50 | area=all | maxDets=100] = 0.405 AP@[IoU=0.75 | area=all | maxDets=100] = 0.221 That’s pretty good! And we can compare the obtained APs with the SSD MobileNet v2 320x320 mAP as from the COCO Dataset documentation:

We make no distinction between AP and mAP (and likewise AR and mAR) and assume the difference is clear from context.

The Average Recall(AR) was split by the max number of detection per image (1, 10, 100). When we have just one kangaroo per image, the recall is around 30% while when we have up to 100 kangaroos it is around 51%. These values are not that good but are reasonable for the kind of problem we’re trying to solve.

(AR)@[ IoU=0.50:0.95 | area=all | maxDets= 1] = 0.293 (AR)@[ IoU=0.50:0.95 | area=all | maxDets= 10] = 0.414 (AR)@[ IoU=0.50:0.95 | area=all | maxDets=100] = 0.514 The Loss analysis is very straightforward, we’ve got 4 values:

INFO:tensorflow: + Loss/localization_loss: 0.345804 INFO:tensorflow: + Loss/classification_loss: 1.496982 INFO:tensorflow: + Loss/regularization_loss: 0.130125 INFO:tensorflow: + Loss/total_loss: 1.972911 The localization loss computes the difference between the predicted bounding boxes and the labeled ones. The classification loss indicates whether the bounding box class matches with the predicted class. The regularization loss is generated by the network’s regularization function and helps to drive the optimization algorithm in the right direction. The last term is the total loss and is the sum of three previous ones.

Tensorflow provides a tool to visualize all these metrics in an easy way. It’s called TensorBoard and can be initialized by the following command:

%load_ext tensorboard 
%tensorboard --logdir {path_root}'/training/' 
#%reload_ext tensorboard #reload it after started
This is going to be shown, and you can explore all training and evaluation metrics.
Tensorboard — Loss

In the tab IMAGES, it’s possible to find some comparisons between the predictions and the ground truth side by side. A very interesting resource to explore during the validation process as well.

Tensorboard — Testing images

Exporting the model
Now that the training is validated, it’s time to export the model. We’re going to convert the training checkpoints to a protobuf (pb) file. This file is going to have the graph definition and the weights of the model.

Caution: deletes inference_graph exported model folder - Run it if you want to export the model again.
#!rm -r {path_root}/inference_graph
Export the Inference Graph
The below code cell adds a line to the tf_utils.py file. This is a temporary fix to a exporting issue occuring when using the API with Tensorflow 2. This code will be removed as soon as the TF Team puts out a fix.

All credit goes to the Github users Jacobsolawetz and Tanner Gilbert, who provided this temporary fix.

!wget -P {path_root}/ https://github.com/bighuang624/Andrew-Ng-Deep-Learning-notes/blob/master/assignments/assignment2-3/tf_utils.py
--2022-06-28 05:24:42--  https://github.com/bighuang624/Andrew-Ng-Deep-Learning-notes/blob/master/assignments/assignment2-3/tf_utils.py
Resolving github.com (github.com)... 140.82.114.3
Connecting to github.com (github.com)|140.82.114.3|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: unspecified [text/html]
Saving to: ‘/content/drive/MyDrive/DSs/glasses_detection_project/tf_utils.py’

tf_utils.py             [ <=>                ] 194.86K  --.-KB/s    in 0.05s   

2022-06-28 05:24:42 (4.00 MB/s) - ‘/content/drive/MyDrive/DSs/glasses_detection_project/tf_utils.py’ saved [199538]

with open(path_root+'/tf_utils.py') as f:
    tf_utils = f.read()

with open(path_root+'/tf_utils.py', 'w') as f:
  # Set labelmap path
  throw_statement = "raise TypeError('Expected Operation, Variable, or Tensor, got ' + str(x))"
  tf_utils = tf_utils.replace(throw_statement, "if not isinstance(x, str):" + throw_statement)
  f.write(tf_utils)
output_directory = path_root+'/inference_graph'

!python {path_root}/models/research/object_detection/exporter_main_v2.py \
    --trained_checkpoint_dir {model_dir} \
    --output_directory {output_directory} \
    --pipeline_config_path {pipeline_config_path} 
2022-06-28 13:52:30.323223: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
WARNING:tensorflow:From /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py:464: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with back_prop=False is deprecated and will be removed in a future version.
Instructions for updating:
back_prop=False is deprecated. Consider using tf.stop_gradient instead.
Instead of:
results = tf.map_fn(fn, elems, back_prop=False)
Use:
results = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))
W0628 13:52:30.486592 139704343304064 deprecation.py:619] From /usr/local/lib/python3.7/dist-packages/tensorflow/python/autograph/impl/api.py:464: calling map_fn_v2 (from tensorflow.python.ops.map_fn) with back_prop=False is deprecated and will be removed in a future version.
Instructions for updating:
back_prop=False is deprecated. Consider using tf.stop_gradient instead.
Instead of:
results = tf.map_fn(fn, elems, back_prop=False)
Use:
results = tf.nest.map_structure(tf.stop_gradient, tf.map_fn(fn, elems))
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:52:36.060364 139704343304064 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:52:36.060756 139704343304064 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:52:36.060979 139704343304064 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:52:36.061187 139704343304064 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:52:36.061388 139704343304064 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
INFO:tensorflow:depth of additional conv before box predictor: 0
I0628 13:52:36.061625 139704343304064 convolutional_keras_box_predictor.py:153] depth of additional conv before box predictor: 0
WARNING:tensorflow:Skipping full serialization of Keras layer <object_detection.meta_architectures.ssd_meta_arch.SSDMetaArch object at 0x7f0ed00eca50>, because it is not built.
W0628 13:52:44.703764 139704343304064 save_impl.py:72] Skipping full serialization of Keras layer <object_detection.meta_architectures.ssd_meta_arch.SSDMetaArch object at 0x7f0ed00eca50>, because it is not built.
2022-06-28 13:52:54.544986: W tensorflow/python/util/util.cc:368] Sets are not currently considered sequences, but this may change in the future, so consider avoiding using them.
W0628 13:53:09.203552 139704343304064 save.py:268] Found untraced functions such as BoxPredictor_layer_call_fn, BoxPredictor_layer_call_and_return_conditional_losses, BoxPredictor_layer_call_fn, BoxPredictor_layer_call_and_return_conditional_losses, BoxPredictor_layer_call_and_return_conditional_losses while saving (showing 5 of 125). These functions will not be directly callable after loading.
INFO:tensorflow:Assets written to: /content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/saved_model/assets
I0628 13:53:14.010262 139704343304064 builder_impl.py:784] Assets written to: /content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/saved_model/assets
INFO:tensorflow:Writing pipeline config file to /content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/pipeline.config
I0628 13:53:14.631900 139704343304064 config_util.py:254] Writing pipeline config file to /content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/pipeline.config
As we’re going to deploy the model using TensorFlow.js and Google Colab has a maximum lifetime limit of 12 hours, let’s download the trained weights and save them locally. When running the command files.download('/< your_project_folder >/saved_model.zip"), the colab will prompt the file automatically.

Compressing the tensorflow saved model to download - Run once if you are using Google drive else once per session.
!zip -r {path_root}/inference_graph/saved_model.zip {path_root}/inference_graph/saved_model
  adding: content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/saved_model/ (stored 0%)
  adding: content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/saved_model/variables/ (stored 0%)
  adding: content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/saved_model/variables/variables.data-00000-of-00001 (deflated 7%)
  adding: content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/saved_model/variables/variables.index (deflated 77%)
  adding: content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/saved_model/assets/ (stored 0%)
  adding: content/drive/MyDrive/DSs/glasses_detection_project/inference_graph/saved_model/saved_model.pb (deflated 92%)
Downloading the saved zipped model
from google.colab import files
files.download(path_root+"/inference_graph/saved_model.zip")
If you want to check if the model was saved properly, load, and test it. I’ve created some functions to make this process easier so feel free to clone the inferenceutils.py file from my GitHub to test some images.
Testing the trained model
Based on Object Detection API Demo and Inference from saved model tf2 colab.

Run only once if you are using google drive or once per session
!wget -P {path_root}/ https://raw.githubusercontent.com/hugozanini/object-detection/master/inferenceutils.py
--2022-06-28 05:17:18--  https://raw.githubusercontent.com/hugozanini/object-detection/master/inferenceutils.py
Resolving raw.githubusercontent.com (raw.githubusercontent.com)... 185.199.108.133, 185.199.109.133, 185.199.110.133, ...
Connecting to raw.githubusercontent.com (raw.githubusercontent.com)|185.199.108.133|:443... connected.
HTTP request sent, awaiting response... 200 OK
Length: 2219 (2.2K) [text/plain]
Saving to: ‘/content/drive/MyDrive/DSs/glasses_detection_project/inferenceutils.py’

inferenceutils.py   100%[===================>]   2.17K  --.-KB/s    in 0.001s  

2022-06-28 05:17:18 (3.15 MB/s) - ‘/content/drive/MyDrive/DSs/glasses_detection_project/inferenceutils.py’ saved [2219/2219]

Loading the model
from inferenceutils import *
output_directory = path_root+'/inference_graph'
category_index = label_map_util.create_category_index_from_labelmap(labelmap_path, use_display_name=True)
tf.keras.backend.clear_session()
model = tf.saved_model.load(f'{output_directory}/saved_model/')
Selecting the images to test
import pandas as pd
test = pd.read_csv(path_root+'/dataset/test_labels.csv')
#Getting 3 random images to test
images = list(test.sample(n=3)['filename'])
Making inferences
for image_name in images:
  
  image_np = load_image_into_numpy_array(path_root+'/dataset/images/' + image_name)
  output_dict = run_inference_for_single_image(model, image_np)
  vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      output_dict['detection_boxes'],
      output_dict['detection_classes'],
      output_dict['detection_scores'],
      category_index,
      instance_masks=output_dict.get('detection_masks_reframed', None),
      use_normalized_coordinates=True,
      line_thickness=3)
  display(Image.fromarray(image_np))



Everything is working well, so we’re ready to put the model in production.
Deploying the model
The model is going to be deployed in a way that anyone can open a PC or mobile camera and perform inferences in real-time through a web browser. To do that, we’re going to convert the saved model to the Tensorflow.js layers format, load the model in a javascript application and make everything available on Glitch.

Converting the model
At this point, you should have something similar to this structure saved locally:

├── inference-graph
│ ├── saved_model
│ │ ├── assets
│ │ ├── saved_model.pb
│ │ ├── variables
│ │ ├── variables.data-00000-of-00001
│ │ └── variables.index
Before we start, let's create an isolated Python environment to work in an empty workspace and avoid any library conflict. Install virtualenv and then open a terminal in the inference-graph folder and create and activate a new virtual environment:
virtualenv -p python3 venv
source venv/bin/activate
Install the TensorFlow.js converter:
pip install tensorflowjs[wizard]
Start the conversion wizard:
tensorflowjs_wizard
Now, the tool will guide you through the conversion, providing explanations for each choice you need to make. The image below shows all the choices that were made to convert the model. Most of them are the standard ones, but options like the shard sizes and compression can be changed according to your needs.

To enable the browser to cache the weights automatically, it’s recommended to split them into shard files of around 4MB. To guarantee that the conversion is going to work, don’t skip the op validation as well, not all TensorFlow operations are supported so some models can be incompatible with TensorFlow.js — See this list for which ops are currently supported.

Installing tensorflowjs
!pip3 install tensorflowjs
Looking in indexes: https://pypi.org/simple, https://us-python.pkg.dev/colab-wheels/public/simple/
Requirement already satisfied: tensorflowjs in /usr/local/lib/python3.7/dist-packages (3.18.0)
Requirement already satisfied: packaging~=20.9 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (20.9)
Requirement already satisfied: six<2,>=1.12.0 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (1.15.0)
Requirement already satisfied: tensorflow-hub<0.13,>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (0.12.0)
Requirement already satisfied: tensorflow<3,>=2.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflowjs) (2.8.2+zzzcolab20220527125636)
Requirement already satisfied: pyparsing>=2.0.2 in /usr/local/lib/python3.7/dist-packages (from packaging~=20.9->tensorflowjs) (3.0.9)
Requirement already satisfied: tensorflow-io-gcs-filesystem>=0.23.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (0.26.0)
Requirement already satisfied: h5py>=2.9.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (3.1.0)
Requirement already satisfied: keras-preprocessing>=1.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.1.2)
Requirement already satisfied: termcolor>=1.1.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.1.0)
Requirement already satisfied: flatbuffers>=1.12 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (2.0)
Requirement already satisfied: wrapt>=1.11.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.14.1)
Requirement already satisfied: astunparse>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.6.3)
Requirement already satisfied: tensorflow-estimator<2.9,>=2.8 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (2.8.0)
Requirement already satisfied: numpy>=1.20 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.21.6)
Requirement already satisfied: protobuf<3.20,>=3.9.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (3.17.3)
Requirement already satisfied: keras<2.9,>=2.8.0rc0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (2.8.0)
Requirement already satisfied: tensorboard<2.9,>=2.8 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (2.8.0)
Requirement already satisfied: google-pasta>=0.1.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (0.2.0)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.46.3)
Requirement already satisfied: gast>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (0.5.3)
Requirement already satisfied: libclang>=9.0.1 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (14.0.1)
Requirement already satisfied: typing-extensions>=3.6.6 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (4.2.0)
Requirement already satisfied: opt-einsum>=2.3.2 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (3.3.0)
Requirement already satisfied: absl-py>=0.4.0 in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (1.0.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.7/dist-packages (from tensorflow<3,>=2.1.0->tensorflowjs) (57.4.0)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /usr/local/lib/python3.7/dist-packages (from astunparse>=1.6.0->tensorflow<3,>=2.1.0->tensorflowjs) (0.37.1)
Requirement already satisfied: cached-property in /usr/local/lib/python3.7/dist-packages (from h5py>=2.9.0->tensorflow<3,>=2.1.0->tensorflowjs) (1.5.2)
Requirement already satisfied: google-auth<3,>=1.6.3 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.35.0)
Requirement already satisfied: google-auth-oauthlib<0.5,>=0.4.1 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (0.4.6)
Requirement already satisfied: markdown>=2.6.8 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (3.3.7)
Requirement already satisfied: requests<3,>=2.21.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (2.23.0)
Requirement already satisfied: werkzeug>=0.11.15 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.0.1)
Requirement already satisfied: tensorboard-data-server<0.7.0,>=0.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (0.6.1)
Requirement already satisfied: tensorboard-plugin-wit>=1.6.0 in /usr/local/lib/python3.7/dist-packages (from tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.8.1)
Requirement already satisfied: cachetools<5.0,>=2.0.0 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (4.2.4)
Requirement already satisfied: rsa<5,>=3.1.4 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (4.8)
Requirement already satisfied: pyasn1-modules>=0.2.1 in /usr/local/lib/python3.7/dist-packages (from google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (0.2.8)
Requirement already satisfied: requests-oauthlib>=0.7.0 in /usr/local/lib/python3.7/dist-packages (from google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.3.1)
Requirement already satisfied: importlib-metadata>=4.4 in /usr/local/lib/python3.7/dist-packages (from markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (4.11.4)
Requirement already satisfied: zipp>=0.5 in /usr/local/lib/python3.7/dist-packages (from importlib-metadata>=4.4->markdown>=2.6.8->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (3.8.0)
Requirement already satisfied: pyasn1<0.5.0,>=0.4.6 in /usr/local/lib/python3.7/dist-packages (from pyasn1-modules>=0.2.1->google-auth<3,>=1.6.3->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (0.4.8)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (1.24.3)
Requirement already satisfied: chardet<4,>=3.0.2 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (2022.5.18.1)
Requirement already satisfied: idna<3,>=2.5 in /usr/local/lib/python3.7/dist-packages (from requests<3,>=2.21.0->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (2.10)
Requirement already satisfied: oauthlib>=3.0.0 in /usr/local/lib/python3.7/dist-packages (from requests-oauthlib>=0.7.0->google-auth-oauthlib<0.5,>=0.4.1->tensorboard<2.9,>=2.8->tensorflow<3,>=2.1.0->tensorflowjs) (3.2.0)
Converting and saving tf saved model into tfjs web model
!tensorflowjs_converter \
     --input_format=tf_saved_model  \
     {path_root}/inference_graph/saved_model \
     {path_root}/inference_graph/web_model
2022-06-08 08:15:02.452021: W tensorflow/core/common_runtime/gpu/gpu_bfc_allocator.cc:39] Overriding allow_growth setting because the TF_FORCE_GPU_ALLOW_GROWTH environment variable is set. Original config value was 0.
Writing weight file /content/drive/MyDrive/DSs/FabrikObjectDetection/inference_graph/saved_model_tfjs/model.json...
If everything worked well, you’re going to have the model converted to the Tensorflow.js layers format in the web_modeldirectory. The folder contains a model.json file and a set of sharded weights files in a binary format. The model.json has both the model topology (aka "architecture" or "graph": a description of the layers and how they are connected) and a manifest of the weight files (Lin, Tsung-Yi, et al).

└ web_model
  ├── group1-shard1of5.bin
  ├── group1-shard2of5.bin
  ├── group1-shard3of5.bin
  ├── group1-shard4of5.bin
  ├── group1-shard5of5.bin
  └── model.json
Compressing tfjs/web model to download - Run once if you are using Google drive else once per session.
!zip -r {path_root}/inference_graph/web_model.zip {path_root}/inference_graph/web_model
  adding: content/drive/MyDrive/DSs/FabrikObjectDetection/inference_graph/web_model/ (stored 0%)
  adding: content/drive/MyDrive/DSs/FabrikObjectDetection/inference_graph/web_model/group1-shard1of5.bin (deflated 7%)
  adding: content/drive/MyDrive/DSs/FabrikObjectDetection/inference_graph/web_model/group1-shard2of5.bin (deflated 7%)
  adding: content/drive/MyDrive/DSs/FabrikObjectDetection/inference_graph/web_model/group1-shard3of5.bin (deflated 7%)
  adding: content/drive/MyDrive/DSs/FabrikObjectDetection/inference_graph/web_model/group1-shard4of5.bin (deflated 6%)
  adding: content/drive/MyDrive/DSs/FabrikObjectDetection/inference_graph/web_model/group1-shard5of5.bin (deflated 7%)
  adding: content/drive/MyDrive/DSs/FabrikObjectDetection/inference_graph/web_model/model.json (deflated 96%)
Downloading Shards i.e. Converted web tfjs model
from google.colab import files
files.download(path_root+"/inference_graph/web_model.zip")
Extracting folder structure from given github link
!unzip {path_root}"/TFJS-object-detection-master" -d {path_root}"/"
Archive:  /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master.zip
   creating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/.gitignore  
   creating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/git_media/
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/LICENSE  
   creating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/models/
   creating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/models/web_model/
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/models/web_model/group1-shard1of5.bin  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/models/web_model/group1-shard2of5.bin  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/models/web_model/group1-shard3of5.bin  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/models/web_model/group1-shard4of5.bin  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/models/web_model/group1-shard5of5.bin  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/models/web_model/model.json  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/package-lock.json  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/package.json  
   creating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/public/
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/public/index.html  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/README.MD  
   creating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/src/
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/src/index.js  
  inflating: /content/drive/MyDrive/DSs/FabrikObjectDetection/TFJS-object-detection-master/TFJS-object-detection-master/src/styles.css  
Configuring the application
The model is ready to be loaded in javascript. I’ve created an application to perform inferences directly from the browser. Let’s clone the repository to figure out how to use the converted model in real-time. This is the project structure:

├── models
│   └── kangaroo-detector
│       ├── group1-shard1of5.bin
│       ├── group1-shard2of5.bin
│       ├── group1-shard3of5.bin
│       ├── group1-shard4of5.bin
│       ├── group1-shard5of5.bin
│       └── model.json
├── package.json
├── package-lock.json
├── public
│   └── index.html
├── README.MD
└── src
    ├── index.js
    └── styles.css
For the sake of simplicity, I already provide a converted kangaroo-detector model in the models folder. However, let’s put the web_model generated in the previous section in the models folder and test it.

The first thing to do is to define how the model is going to be loaded in the function load_model (lines 10–15 in the file src>index.js). There are two choices.

The first option is to create an HTTP server locally that will make the model available in a URL allowing requests and be treated as a REST API. When loading the model, TensorFlow.js will do the following requests:

GET /model.json
GET /group1-shard1of5.bin
GET /group1-shard2of5.bin
GET /group1-shard3of5.bin
GET /group1-shardo4f5.bin
GET /group1-shardo5f5.bin
If you choose this option, define the load_model function as follows:

  async function load_model() {
    // It's possible to load the model locally or from a repo
    // You can choose whatever IP and PORT you want in the "http://127.0.0.1:8080/model.json"     just set it before in your https server
    const model = await loadGraphModel("http://127.0.0.1:8080/model.json");
    //const model = await loadGraphModel("https://raw.githubusercontent.com/hugozanini/TFJS-object-detection/master/models/web_model/model.json");
    return model;
}
Then install the http-server:

npm install http-server -g
Go to models > web_model and run the command below to make the model available at http://127.0.0.1:8080 . This a good choice when you want to keep the model weights in a safe place and control who can request inferences to it. The -c1 parameter is added to disable caching, and the --cors flag enables cross origin resource sharing allowing the hosted files to be used by the client side JavaScript for a given domain.

http-server -c1 --cors .
Alternatively you can upload the model files somewhere, in my case, I chose my own Github repo and referenced to the model.json URL in the load_model function:

async function load_model() {
    // It's possible to load the model locally or from a repo
    //const model = await loadGraphModel("http://127.0.0.1:8080/model.json");
    const model = await loadGraphModel("https://raw.githubusercontent.com/hugozanini/TFJS-object-detection/master/models/web_model/model.json");
    return model;
}
This is a good option because it gives more flexibility to the application and makes it easier to run on some platform as Glitch.

Running locally
To run the app locally, install the required packages:
npm install
And start:
npm start
The application is going to run at http://localhost:3000 and you should see something similar to this: enter your screenshot image The model takes from 1 to 2 seconds to load and, after that, you can show kangaroos images to the camera and the application is going to draw bounding boxes around them.

Publishing in Glitch
Glitch is a simple tool for creating web apps where we can upload the code and make the application available for everyone on the web. Uploading the model files in a GitHub repo and referencing to them in the load_model function, we can simply log into Glitch, click on New project > Import from Github and select the app repository.

Wait some minutes to install the packages and your app will be available in a public URL. Click on Show > In a new window and a tab will be open. Copy this URL and past it in any web browser (PC or Mobile) and your object detection will be ready to run. See some examples in the video below:

https://youtu.be/WkW50Jn2R5c
First, I did a test showing a kangaroo sign to verify the robustness of the application. It showed that the model is focusing specifically on the kangaroo features and did not specialize in irrelevant characteristics that were present in many images, such as pale colors or shrubs.

Then, I opened the app on my mobile and showed some images from the test set. The model runs smoothly and identifies most of the kangaroos. If you want to test my live app, it is available here (glitch takes some minutes to wake up).

Besides the accuracy, an interesting part of these experiments is the inference time — everything runs in real-time in the browser via JavaScript. Good object detection models running in the browser and using few computational resources is a must in many applications, mostly in industry. Putting the Machine Learning model on the client-side means cost reduction and safer applications as user privacy is preserved as there is no need to send the information to any server to perform the inference.

Next steps:
Object detection in the browser can solve a lot of real-world problems and I hope this article will serve as a basis for new projects involving Computer Vision, Python, TensorFlow and Javascript.

As the next steps, I’d like to make more detailed training experiments. Due to the lack of resources, I could not try many different parameters and I’m sure that there is a lot of room for improvements in the model.

I’m more focused on the models' training, but I’d like to see a better user interface for the app. If someone is interested in contributing to the project, feel free to create a pull request in the project repo. It will be nice to make a more user-friendly application.

If you have any questions or suggestions you can reach me out on Linkedin https://www.linkedin.com/in/hugozanini/?locale=en_US. Thanks for reading!

https://youtu.be/3XzQQlh_p1c
