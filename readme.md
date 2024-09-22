```bash
```


Installazione prerequisiti
```bash
sudo apt update;
sudo apt install -y build-essential libssl-dev zlib1g-dev \
libbz2-dev libreadline-dev libsqlite3-dev curl git \
libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev

#installazione di pyenv
curl https://pyenv.run | bash
```

Setup di pyenv, riavviare il terminale dopo l'esecuzione
```bash
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
echo 'command -v pyenv >/dev/null || export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init -)"' >> ~/.bashrc
```




```bash
git clone https://github.com/SIJ82/STT_whisper.git ./stt
cd stt/
#python -m pip install virtualenv
```

Preparazione dell'ambiente python (ricorda di attivare l'ambiente quando avvi una nuova sessione con "pyenv activate stt")
```bash
pyenv install 3.12.5
pyenv virtualenv 3.12.5 stt
pyenv activate stt
```

```bash
pip install jiwer #for error recognition
pip install faster-whisper
```

```bash
sudo mkrid /models
sudo chmod 777 /models
python install_all_models.py
```


Per verificare che tutto sia installato correttamente si può eseguire un test veloce con il modello tiny
```bash
python fast_test.py
```


pyenv version

pyenv install 3.12.5

#.venv/Scripts/activate

pip install faster-whisper