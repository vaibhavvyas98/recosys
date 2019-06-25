# recommendersys
To build the docker image navigate to the current directory from docker quickstart terminal and enter
  >docker build -t newflaskapp:latest .
This will build an image installing the required dependencies on a base image and it copies the source code into the /var/www/html and also start a Flask development server exposing port 8080 to the host machine.
To run a container from the created image enter
  >docker run -it -p 6000:8080 newflaskapp
Check the docker container IP using 
  >docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' <container_name_or_id>
Go to the browser and enter the IP followed by the port 6000.
The index.html page will open up and your recommender system is up and running.
