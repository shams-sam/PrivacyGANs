docker run --rm -d --name eigan_devel --gpus all --ipc=host -p 8888:8888 -p 6006:6006 -v $(pwd):/WorkSpace dev-py36:latest jupyter notebook --ip=0.0.0.0 --allow-root --no-browser
