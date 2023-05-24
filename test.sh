PROF="/home/nfs_data/zhanggh/ChatGLM-X/profiles"
BENCH="/home/nfs_data/zhanggh/ChatGLM-X/bench"
nsys nvprof -o $PROF/$1 --profile-from-start=on python3 $BENCH/$2.py