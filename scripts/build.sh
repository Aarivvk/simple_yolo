docker run --rm --net=host -it -v$(pwd):/project -w/project --runtime=nvidia --gpus all cpptorch:latest "./scripts/compile"
