# interact -n 20 -t 01:00:00 -m 10g -p 3090-gcondo
streamlist_host=$(hostname -i)
streamlist_port=8501
echo "ssh -N -L ${streamlist_port}:${streamlist_host}:${streamlist_port} ${USER}@sshcampus.ccv.brown.edu"
echo "Paste the following URL into your browser: http://localhost:${streamlist_port}"
output = $(python3 -m streamlit run $CURVATURE_HOME/src/streamlit_app.py --server.port $streamlist_port --server.fileWatcherType none)