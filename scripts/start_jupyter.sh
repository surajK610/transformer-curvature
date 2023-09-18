set XDG_RUNTIME_DIR
interact -n 20 -t 01:00:00 -m 10g -p 3090-gcondo
module load python/3.9.0 cuda/11.1.1 gcc/10.2
source venv/bin/activate
ipnip=$(hostname -i)
ipnport=8889
echo "Paste the following command onto your local computer:"
echo "ssh -N -L ${ipnport}:${ipnip}:${ipnport} sanand14@sshcampus.ccv.brown.edu"
output = $(jupyter-notebook --no-browser --port=$ipnport --ip=$ipnip)

