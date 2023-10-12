To create the docker image, run the following command from the root of the repository
```
sudo docker build -t vllm . 
```

To run the docker image, run the following command from the root of the repository
```
sudo docker run --gpus all -it --rm -p 5000:5000 -e HUGGINGFACE_KEY=$HUGGINGFACE_KEY vllm
```