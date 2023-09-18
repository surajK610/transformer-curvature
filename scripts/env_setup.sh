module load python/3.9.0 cuda/11.1.1 gcc/10.2
python3 -m venv venv
source venv/bin/activate
pip3 install torch transformers rich ftfy regex tqdm pytorch-lightning pandas seaborn pytest matplotlib scipy scikit-learn notebook openai datasets nltk accelerate tensorboardX urllib3==1.26.6 protobuf==3.20.3 

