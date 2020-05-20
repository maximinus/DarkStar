Tensorflow Docker Instructions
==============================

Create a docker container, and run inside as user:

	docker run -u $(id -u):$(id -g) --gpus all -it libs bash

|

We need to mount a directory that we can edit from outside the container:

	docker run -u $(id -u):$(id -g) --gpus all -it -v /home/sparky/code/mlmount:/app tensorflow/tensorflow:2.0.1-gpu-py3 bash

|

Building a container
====================

Build a new container, run and mount image:

	docker build -t {name} .
	docker run -u $(id -u):$(id -g) --gpus all -it -v /home/sparky/code/machine-learning/mount:/app {name} bash

Second time, exec instead:

	docker start {container_name}
	docker exec -it {container_name} bash

|

Running Code Inside a Docker container
======================================

1: Build a container as shown above

2: You can view current machines with

	docker ps -a

3: In our case, the container we will use for our examples is called "machinel"

4: Now start the machine:

	docker start machine1
	docker exec -it machine1 bash

5: You should now be inside a bash console. The mounted data is at /app. You can move files here whilst docker is running.

6: Inside /app, run the code as you would on the main machine.

7: Use tools on the base machine to edit the source code. The docker image should only be for running the ml code.

|
