pwd
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
apt-get install wget
sudo apt-get install wget
wget https://repo.anaconda.com/archive/Anaconda3-2022.10-Linux-x86_64.sh
sudo sh Anaconda3-2022.10-Linux-x86_64.sh 
conda init
cat /root/.bashrc
sudo cat /root/.bashrc
nano 
nano /home/andresr/.bashrc 
conda
mkdir /opt/haystack
sudo mkdir /opt/haystack
chmod 777 -R /opt/haystack/
sudo chmod 777 -R /opt/haystack/
conda create --name haystack
conda activate haystack
cd /opt/haystack/
ls
git clone https://github.com/deepset-ai/haystack.git
apt-get install git
sudo apt-get install git
ls
git clone https://github.com/deepset-ai/haystack.git
cd haystack/
ls
ipython
pip install ipython
pip3
pip
conda
python
which python
sudo apt install python3 python3-pip
pip 
pip install ipython
ipython
python
python
conda activate haystack
ipython
nvidia-smi
lspci 
sudo apt-get install lspci 
sudo update-pciids
sudo apt-get install pciutils 
lspci -v
lsb_release -a
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1804-10-1-local-10.1.243-418.87.00_1.0-1_amd64.deb
sudo apt-key add /var/cuda-repo-10-1-local-10.1.243-418.87.00/7fa2af80.pub
sudo apt-get update
sudo apt-get -y install cuda
sudo add-apt-repository universe
sudo apt-get install freeglut3-dev
sudo apt-get -y install cuda cuda-10-1 cuda-toolkit-10-1 cuda-samples-10-1 cuda-documentation-10-1
sudo apt-get install cuda
apt-get remove --purge nvidia
sudo apt-get remove --purge nvidia
sudo apt-get remove --purge cuda
ls
pwd
git clone https://github.com/GoogleCloudPlatform/compute-gpu-installation.git
ls
cd compute-gpu-installation/
cd linux/
ls
sudo python3 install_gpu_driver.py
nvidia-smi
ipython
pip3 install torch 
ipyhon
import torch
ipython
cat /home/andresr/.bash_history 
l;s
ls
cd /opt/haystack/
ls
cd haystack/
ls
cd ..
mv /opt/haystack /opt/bioasq
sudo mv /opt/haystack /opt/bioasq
cd bioasq/
LS
cd haystack/
ls
pip install -e '.[all-gpu]'
cd haystack/
ls
pip install -e '.[all-gpu]'
chmod -R 777 /opt/bioasq
sudo chmod -R 777 /opt/bioasq
pip install -e '.[all-gpu]'
ls
cd..
ls
cd ..
rm haystack/
rm -rf  haystack
git clone https://github.com/deepset-ai/haystack.git
cd haystack
pip install -e '.[all-gpu]'
chown andresr:andresr /opt/bioasq/
sudo chown andresr:andresr /opt/bioasq/
sudo chown andresr:andresr /opt/bioasq/haystack/
pip install -e '.[all-gpu]'
pip install --upgrade pip
pip install -e '.[all-gpu]' 
cd ..
mkdir resources
wget http://participants-area.bioasq.org/info/BioASQword2vec/
ls
rm index.html 
cd resources/
http://participants-area.bioasq.org/tools/BioASQword2vec/biomedicalWordVectors.tar.gz
wget http://participants-area.bioasq.org/tools/BioASQword2vec/biomedicalWordVectors.tar.gz
ls
mkdir pubmed_baseline_2023
cd pubmed_baseline_2023/
wget -A gz,md5 -m -p -E -k -K -np https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
wget -m -p -E -k -K -np https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
ls
rm ftp.ncbi.nlm.nih.gov/
curl https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
ls
rm -rf ftp.ncbi.nlm.nih.gov
wget --execute="robots = off" --mirror --convert-links --no-parent --wait=5 https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
screen -S haystack
apt-get install screen
sudo apt-get install screen
screen -S download
screen -R download
screen -L
screen -ls
screen -r download
cd /opt/
wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
ls
sudo wget https://bin.equinox.io/c/bNyj1mQVY4c/ngrok-v3-stable-linux-amd64.tgz
ls
sudo tar xvzf ngrok-v3-stable-linux-amd64.tgz -C /usr/local/bin
ngronk
ngrok
ngrok http 80
ngrok config add-authtoken 5QzR69ttW8RnhFCWVYLBz_7qTDTT3EcZcnMQhPUvXUZ
conda activate haystack
pip install notebook
conda install -c conda-forge notebook
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade openssl
python3 -m pip install openssl>22.1.0
sudo python3 -m pip install openssl>22.1.0
jupyter notebook
jupyter notebook --generate-config
nano /home/andresr/.jupyter/jupyter_notebook_config.py
jupyter notebook
nano /home/andresr/.jupyter/jupyter_notebook_config.py
conda activate haystack
nano .jupyter/jupyter_notebook_config.py 
cd /opt/bioasq/haystack/
ls
nano docker-compose.yml 
ls
sudo apt -y install apt-transport-https ca-certificates curl gnupg2 software-properties-common
curl -fsSL https://download.docker.com/linux/debian/gpg | sudo gpg --dearmor -o /etc/apt/trusted.gpg.d/docker-archive-keyring.gpg
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io -y
sudo add-apt-repository    "deb [arch=amd64] https://download.docker.com/linux/debian \
   $(lsb_release -cs) \
   stable"
sudo apt install docker-ce docker-ce-cli containerd.io -y
sudo apt update
sudo apt install docker-ce docker-ce-cli containerd.io -y
sudo systemctl enable --now docker
sudo usermod -aG docker $USER
newgrp docker
docker version
docker-compose up
docker version
curl -s https://api.github.com/repos/docker/compose/releases/latest | grep browser_download_url  | grep docker-compose-linux-x86_64 | cut -d '"' -f 4 | wget -qi -
ls
rm docker-compose-linux-x86_64*
cd ..
curl -s https://api.github.com/repos/docker/compose/releases/latest | grep browser_download_url  | grep docker-compose-linux-x86_64 | cut -d '"' -f 4 | wget -qi -
chmod +x docker-compose-linux-x86_64
sudo mv docker-compose-linux-x86_64 /usr/local/bin/docker-compose
docker-compose version
sudo curl -L https://raw.githubusercontent.com/docker/compose/master/contrib/completion/bash/docker-compose -o /etc/bash_completion.d/docker-compose
source /etc/bash_completion.d/docker-compose
docker-compose ps
docker-compose -f docker-compose.yml up
cd haystack/
docker-compose -f docker-compose.yml up
ngrok 8888
conda activate haystack
ngrok
ngrok 8888
ngrok http 8888
conda activate haystack
nano .jupyter/jupyter_notebook_config.py 
jupyter notebook
nano .jupyter/jupyter_notebook_config.py 
conda activate bioasq
conda activate haystack
jupyter notebook
ipython
screen -S jupyter
ls
rm  -rf ftp.ncbi.nlm.nih.gov
cat /home/andresr/.bash_history 
wget --execute="robots = off" --mirror --convert-links --no-parent --wait=5 https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/
ls
mv ftp.ncbi.nlm.nih.gov/pubmed/baseline/*.gz .
ls
docker-compose ps
docker ps
sudo docker ps
screen -ls
screen -S haystack
cat .bash_history 
cat /home/andresr/scripts/
cat /home/andresr/scripts/init_haystack.sh 
conda activate haystackconda activate haystack
conda activate haystack
nano /home/andresr/scripts/init_haystack.sh 
ls /home/andresr/logs/
cat /home/andresr/logs/haystack_docker.log 
cat /home/andresr/logs/jupyter.log 
ls -lrt /home/andresr/logs/
nano /home/andresr/.bashrc
yq
ls
pwd
mkdir scripts
cd scripts/
nano init_haystack.sh
ls
bash init_haystack.sh 
conda 
sh init_haystack.sh 
bash init_haystack.sh 
mkdir /home/andresr/logs
bash init_haystack.sh 
ls
pwd
cd /opt/bioasq/
git clone https://github.com/andresrosso/col-un-bioasq11.git
screen -list
screen -r 46494.pts-0.nlp-lab
screen -r 46486.pts-0.nlp-lab
ls
screen -ls
screen -l
screen -L
screen -l
screen -list
screen -x 46494.pts-0.nlp-lab
screen -ls
screen -X -S 46*
screen -X -S 46494.pts-0.nlp-lab
screen -X -S 46494.pts-0.nlp-lab quit
screen -X -S 46486.pts-0.nlp-lab quit
screen -X -S 46460.pts-0.nlp-lab quit
screen -ls
screen -S haystack
cd /opt/bioasq/
ls
git clone https://github.com/andresrosso/col-un-bioasq11.git
nano /home/andresr/.git-credentials
git config --global credential.helper store
nano /home/andresr/.git-credentials
git clone https://github.com/andresrosso/col-un-bioasq11.git
cd scripts/
ls
bash init_haystack.sh 
ls 
cd /opt/bioasq/
ls
cd ..
ls
cd bioasq/resources/pubmed_baseline_2023/
ls
ps -fea | grep elastic
cat /opt/bioasq/resources/pubmed_baseline_2023/pubmed23n0502.xml.gz
cd /opt/bioasq/resources/
ls
cd pubmed_baseline_2023/
ls
mkdir test
cp pubmed23n116* test/
curl GET /pubmed2023-2/_count
screen -ls
screen -S haystack
screen -ls
screen -r haystack
df -h
screen -r haystack
ps -fea
htop
apt-get install htop
sudo apt-get install htop
htop
cd /opt/bioasq/col-un-bioasq11/notebooks/elastic_search/
ls
cd ..
cd haystack/
ls
wc -l result.csv 
htop
kill -9 1755
htop
screen -t haystack
htop
runlevel
sudo runlevel
ln -s /home/andresr/scripts/init_haystack.sh /etc/rc5.d/S99init_haystack
sudo ln -s /home/andresr/scripts/init_haystack.sh /etc/rc5.d/S99init_haystack
cd scripts/
nano init_haystack.sh 
bash init_haystack.sh 
screen -S haystack
ls
cd /opt/bioasq/col-un-bioasq11/
git add .
git commit -m "haystack"
git push
git add .
git status
nano .gitignore 
LS
ls
cd notebooks/haystack/
ls -lrt
wc -l bioasq-indexing.log 
wc -l result.csv 
htop
cd ..
git config --global credential.helper store
git pull
~/.git-credentials
sudo ~/.git-credentials
sudo nano ~/.gitconfig
git add .
git status
git restore --staged notebooks/haystack/result.csv 
git status
git commit -m "haystack"
git config --global user.email "andresrosso@gmail.com"
git config --global user.name andresrosso
git commit -m "haystack"
git push
nano .gitignore 
wc -l notebooks/haystack/result.csv 
df -h
wc -l notebooks/haystack/result.csv 
df -h
wc -l notebooks/haystack/result.csv 
cd /opt/bioasq/
wc -l notebooks/haystack/result.csv 
cd col-un-bioasq11/
wc -l notebooks/haystack/result.csv 
cd ..
git clone https://github.com/andresrosso/Evaluation-Measures.git
ls
git clone https://github.com/andresrosso/dmlpr.git
df -h
wc -l notebooks/haystack/result.csv 
cd col-un-bioasq11/
wc -l notebooks/haystack/result.csv 
cd..
cd ..
ls -lrt
git clone https://github.com/andresrosso/bioasq8-unal.git
git clone https://github.com/andresrosso/bioasq7-tf-models.git
cd col-un-bioasq11/
python globals.py 
conda activate haystack
conda
nano /home/andresr/.bashrc 
source ~/.bashrc
nano /home/andresr/.bashrc 
source ~/.bashrc
nano /home/andresr/.bashrc 
python setup.py 
ls
python globals.py 
cd /opt/bioasq/
ls Evaluation-Measures/examples/aueb_google_docs/aueb_nlp-bioasq6b-submissions/
cd col-un-bioasq11/
git add .
git status
git commit -m "doc retrieval"
git push
cd ..
screen -ls
cat .bash_history 
cat .bash_history | grep ngrok
cat scripts/init_haystack.sh 
cat /home/andresr/logs/ngrok.log 
cat /home/andresr/logs/haystack_docker.log 
screen -S haystack
curl -XGET 'http://loadtest-appserver1:9200/pubmed2023/_mapping'
curl -XGET 'http://localost:9200/pubmed2023/_mapping'
curl -XGET 'http://localhost:9200/pubmed2023/_mapping'
curl -XPOST http://localhost:9200/INDEX_NAME/_update_by_query
{   "query": { ;     "bool": {;         "must_not": {;             "exists": {;                 "field": "abstract";             }
cd /opt/bioasq/resources/pubmed_baseline_2023/test/
htop
nano /home/andresr/.bashrc 
screen -S haystack
htop
cd /opt/bioasq/resources/pubmed_baseline_2023/
ls
ls test/
cp pubmed23n103* test/
htop
cd /home/andresr/scripts/
bash init_haystack.sh 
nano scripts/init_haystack.sh 
nano .bashrc 
screen  -S haystack
screen -S indexing
nano .bashrc 
conda config --set auto_activate_base false
screen -S indexing
source activate base
conda activate base
conda env list
pip3 install pyOpenSSL --upgrade
conda env list
conda activate haystack
screen -R indexing
screen -r indexing
screen -X indexing
screen -r 3228.indexing
screen -S es-index
htop
scree -l
screen -l
screen -ls
screen -a es-index
screen -r es-index
screen -ls
screen -r es-indexing
screen -r es-index
screen -ls
screen -r es-index
cat .bash_history | grep count
curl -XGET /pubmed2023-2/_count
curl -XGET http://loadtest-appserver1:9200/pubmed2023-old/_count
curl -XGET http://localhost:9200/pubmed2023-old/_count
curl -XGET http://localhost:9200/pubmed2023/_count
curl -XGET http://localhost:9200/pubmed2023-old/_count
pwd
ls
mkdir /opt/bioasq/backup
mkdir /opt/bioasq/backup/es-index
curl -XPUT 'http://localhost:9200/_snapshot/bioasq_backup' -d '{
    "type": "fs",
    "settings": {
        "location": "/opt/bioasq/backup/es-index",
        "compress": true
    }
}'
curl -XPUT 'http://localhost:9200/_snapshot/bioasq_backup?verify=false' -d '{
    "type": "fs",
    "settings": {
        "location": "/opt/bioasq/backup/es-index",
        "compress": true
    }
}'
curl -XPUT 'http://localhost:9200/_snapshot/bioasq_backup?verify=false' -H "Content-Type: application/json" -d '{
    "type": "fs",
    "settings": {
        "location": "/opt/bioasq/backup/es-index",
        "compress": true
    }
}'
curl -XPUT 'http://localhost:9200/_snapshot/bioasq_backup' -H "Content-Type: application/json" -d '{
    "type": "fs",
    "settings": {
        "location": "/opt/bioasq/backup/es-index",
        "compress": true
    }
}'
df -h
ls
pwd
cd /opt/bioasq/
ls
cd resources/
ls
mkdir training-data
cd training-data/
wget http://participants-area.bioasq.org/Tasks/11b/trainingDataset/
ls
rm index.html 
lynx http://participants-area.bioasq.org/Tasks/11b/trainingDataset/
apt-get install lynx
sudo apt-get install lynx
lynx http://participants-area.bioasq.org/Tasks/11b/trainingDataset/
mv /opt/bioasq/col-un-bioasq11/data/raw/BioASQ-training11b.zip .
ls
unzip BioASQ-training11b.zip 
tar -xvf BioASQ-training11b.zip 
ls
gzip -xvf BioASQ-training11b.zip 
sudo apt-get install unzip
unzip BioASQ-training11b.zip 
ls
mkdir 11b
mv BioASQ-training* 11b/
ls
cd 11b/
ls
cd BioASQ-training11b
ls
cat training11b.json 
cp training11b.json /opt/bioasq/col-un-bioasq11/data/raw/
ls /opt/bioasq/col-un-bioasq11/data/raw/
screen -ls
screen -r es-index
conda env list
conda create iccp
conda create -n iccp
conda activate iccp
cd /opt/
mkdir iccp
ls
sudo mkdir iccp
cd chmod 777 iccp/
sudo chmod 777 iccp
pip install opencv-python
ipython kernel install --name "iccp" 
ipython kernel install --name "iccp" --user
mkdir iccp/data
sudo apt install tesseract-ocr -y
pip install openpyxl
screen -s haystack
screen -ls
screen -S haystack
conda activate haystack
conda install -c conda-forge jupyterlab
jupyter lab
screen -r haystack
nano /home/andresr/scr
nano /home/andresr/scripts/init_haystack.sh 
ngrok http 8889
nano /home/andresr/scripts/init_haystack.sh 
lynx 34.134.20.18 8888
lynx https://34.134.20.18 8888
wget https://34.134.20.18:8888
lynx https://34.134.20.18:8888
lynx https://34.134.20.18:80
lynx http://34.134.20.18:80
lynx http://localhost:80
lynx https://localhost:8888
netstat -pan 
lynx https://127.0.0.1:8888
lynx https://0.0.0.0:8888
lynx https://0.0.0.0:8884
lynx https://127.0.0.1:8881
lynx https://127.0.0.1:8888
screen -ls
screen -r haystack
nano /home/andresr/.jupyter/jupyter_notebook_config.py 
conda activate haystack
nano /home/andresr/.bashrc 
conda activate base
conda activate haystack
ls
cd /home/andresr/scripts/
ls
bash init_haystack.sh 
screen -r haystack
conda activate haystack
cd /home/andresr/scripts/
ls
bash init_haystack.sh 
cat /home/andresr/scripts/init_haystack.sh 
tail -f /home/andresr/logs/jupyter.log
tail -f /home/andresr/logs/jupyter.log 
cat /home/andresr/logs/jupyter.log 
ls -lrt /home/andresr/logs/jupyter.log 
cat /home/andresr/logs/jupyter.log
nano /home/andresr/logs/jupyter.log
ls -la /home/andresr/logs/jupyter.log
ls -lart /home/andresr/logs/jupyter.log
ls -lrt /home/andresr/logs/jupyter.log
cat /home/andresr/logs/jupyter.log
nano /home/andresr/.jupyter/jupyter_notebook_config.py 
conda activate haystack
jupyter notebook list
jupyter notebook stop 8888
sudo jupyter notebook stop 8888
pkill
sudo apt-get install pkill
ps aux|grep jupyter
kill -9 2927
ps aux|grep jupyter
kill -9 6436
kill -9 5136
ps aux|grep jupyter
kill -9 6229
ps aux|grep jupyter
jupyter notebook start
jupyter-notebook start
jupyter notebook
nano /home/andresr/.jupyter/jupyter_notebook_config.py 
jupyter notebook
nano /home/andresr/.jupyter/jupyter_notebook_config.py 
sudo nano /etc/ssh/sshd_config
sudo service ssh restart
sudo andresr
su -i andresr
su andresr
sudo passwd andresr
su andresr
screen -ls
screen -r jupyter
screen -r 2707.jupyter
screen -d jupyter
screen -r jupyter
sudo ls
exit
sudo ls]
sudo ls
screen -ls
screen -r jupyter
screen -ls
screen -r jupyter
python -m ipykernel install --user --name=haystack
conda activate haystack
screen -r haystack
screen -S haystack
screen -S jupyter
jupyter lab
conda activate haystack
jupyter lab
nano scripts/jupyter.sh 
python -m ipykernel install --user --name=haystack
pip install elasticsearch
pip install nltk
pip install pandas
pip install matplotlib
pip install seaborn
pip install pytrec_eval
ls -lrt /home/
curl -XGET http://loadtest-appserver1:9200/pubmed2023-old/_count
curl -XGET http://localhost:9200/pubmed2023-old/_count
sudo ls
sudo apt-get lsof
usermod -aG sudo andresr
sudo usermod -aG sudo andresr
sudo apt-get lsof
sudo apt-get install lsof
conda activate haystack
nano /home/andresr/.bashrc 
conda activate haystack
nano /home/andresr/.bashrc 
conda activate haystack
nano /home/andresr/.bashrc 
. /opt/anaconda3/etc/profile.d/conda.sh
export PATH="/opt/anaconda3/bin:$PATH"
conda activate haystack
cd /home/andresr/scripts/
ls
bash jupyter.sh 
nano /home/andresr/.jupyter/jupyter_notebook_config.py 
bash jupyter.sh 
nano /home/andresr/.jupyter/jupyter_notebook_config.py 
bash jupyter.sh 
ps -fea | grep jupyter
ps -fea | grep python
jupyter notebook lsi
jupyter notebook list
pggreo
pggrep
jupyter notebook stop
lsof -n -i4TCP:8888
sudo apt-get install lsof
sudo ls
su root
passwd root
sudo passwd 
lsof -n -i4TCP:8888
kill -9 3168
lsof -n -i4TCP:8889
kill -9 4753
lsof -n -i4TCP:8890
kill -9 6780
jupyter notebook list
kill -9 8891
lsof -n -i4TCP:8891
kill -9 6892
jupyter notebook
lsof -n -i4TCP:88888
nano /home/andresr/.jupyter/jupyter_notebook_config.py 
jupyter notebook
. jupyter.sh 
jupyter lab
nano /home/andresr/.jupyter/jupyter_notebook_config.py 
nano jupyter.sh 
jupyter lab > /home/andresr/logs/jupyterlab.log
conda activate haystack
cd /home/andresr/scripts/
bash init_haystack.sh 
ls
nano init_haystack.sh 
ls
nano jupyter.sh
bash init_haystack.sh 
screen -ls
screen -r jupyter
conda activate haystack
curl -XGET http://localhost:9200/_cat/indices
screen -S haystack
screen -S jupyter
cd opt
cd /opt/
ls
cd bioasq/haystack/
ls
ls rest_api/rest_api/pipeline/
ls
mkdir es_data
ls
ls es_data/
docker volume ls
nano docker-compose.yml 
screen -S indexing
conda activate haystack
pip install pytorch_lightning
pip install 'lightning'
conda install pytorch-lightning -c conda-forge
docker volume ls
nano /opt/bioasq/haystack/docker-compose.yml 
screen -ls
screen -r haystack
screen -ls
screen -r indexing
htop
pwd
mkdir /opt/bioasq/data
cd /opt/bioasq/data/
ls
ls 2019_data/
cat 2019_data/generated_train_dataset/4b_to_7b_dataset/511a20f3df1ebcce7d00000c.json 
cd ..
ls
git pull https://github.com/andresrosso/metric-learning-bert.git
git glone https://github.com/andresrosso/metric-learning-bert.git
git clone https://github.com/andresrosso/metric-learning-bert.git
conda activate haystack
cd ..
find bioasq -name samples.json
find bioasq -name poc_ml.ipynb
find bioasq -name samples.json
df -h
ls -lrt /opt/bioasq/haystack/es_data/nodes/0/indices/
conda activate haystack
python
cd /opt/bioasq/
git clone https://github.com/GoogleCloudPlatform/compute-gpu-installation.git
sudo python3 install_gpu_driver.py
cd compute-gpu-installation/
sudo python3 install_gpu_driver.py
ls
cd linux/
ls
sudo python3 install_gpu_driver.py
pip3 install torch torchvision torchaudio
python
screen -ls
screen -r indexing
screen -d indexing
screen -r indexing
screen -ls
screen -r indexing
clear
cd /opt/bioasq/
tar xvf metric-learning-bert
tar xvf metric-learning-bert.tgz 
cd metric-learning-bert
ls
rm -rf metric-learning-bert
ls
ls /opt/bioasq/Evaluation-Measures
ls /opt/bioasq/Evaluation-Measures/working_folder/
nano /opt/bioasq/col-un-bioasq11/doc_retrieval_test_5-aueb-nlp-4.json
conda activate haystack
pip install Haystack
pip install Haystack install git+https://github.com/deepset-ai/haystack.git
pip install farm-haystack
conda env list
conda install -n haystack-py310 python=3.10
conda create --name haystack-py310 python=3.10 anaconda
conda activate haystack-py310
pip install pyqt5
sudo apt install python3 python3-pip
pip install pyqt5
pip install pyqtwebengine
pip install ruamel-yaml
pip install clyent==1.2.1
pip uninstall pydoc-markdown
python3 -m pip install PyYAML==6.0
pip install --upgrade pip
cd /opt/bioasq/haystack/
ls
conda activate haystack
pip install -e '.[all-gpu]'
conda activate haystack-py310
pip install -e '.[all-gpu]'
pip3 install torch torchvision torchaudio
ipython
python -m ipykernel install --user --name=haystack-py310
pip install pytrec_eval
BM25Retriever
ElasticsearchRetriever
DensePassageRetriever
TableTextRetriever
EmbeddingRetriever
TfidfRetriever
ElasticsearchFilterOnlyRetriever
MultiModalRetriever
cd /opt/bioasq/col-un-bioasq11/
git add .
git commit -m "doc retrieveal with haystack"
git push
git add .
git commit -m "doc retrieveal with haystack"
git push
nvidia smi
nvidia-smi 
cd /opt/bioasq/resources/embeddings/
ls -lrt
cp wikipedia-pubmed-and-PMC-w2v.bin.bkp wikipedia-pubmed-and-PMC-w2v.bin
cd /opt/bioasq/
ls
cd resources/
ls
mkdir embeddings
cd embeddings/
wget http://evexdb.org/pmresources/vec-space-models/wikipedia-pubmed-and-PMC-w2v.bin
pwd
cd ..
mv /opt/bioasq/resources/embeddings/wikipedia-pubmed-and-PMC-w2v.bin /opt/bioasq/resources/embeddings/wikipedia-pubmed-and-PMC-w2v.bin.bkp
nvidia-smi
