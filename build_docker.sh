uid=$(id -u $USER)
gid=$(id -g $USER)

docker build -t asd_pytorch_2023 ./ \
    --build-arg USERNAME=$USER \
    --build-arg GROUPNAME=$USER \
    --build-arg UID=$uid \
    --build-arg GID=$gid \
