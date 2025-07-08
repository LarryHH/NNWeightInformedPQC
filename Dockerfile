FROM python:3.11
WORKDIR /usr/src/app
COPY . .
RUN apt-get update && apt-get install -y git wget nano

RUN pip3 install --no-cache-dir -r requirements.txt

CMD ["bash"]

# docker build -t larry:nnwipqc .
# docker run -d --rm -v ./data:/usr/src/app/data -v ./tsne/results:/usr/src/app/tsne/results -v ./pca/results:/usr/src/app/pca/results -v ./custom_clustering_utils/results:/usr/src/app/custom_clustering_utils/results larry:csiro_workstation
# docker run -it --rm -v ./data:/usr/src/app/data -v ./tsne/results:/usr/src/app/tsne/results ./pca/results:/usr/src/app/pca/results -v ./custom_clustering_utils/results:/usr/src/app/custom_clustering_utils/results larry:csiro_workstation

# docker run --gpus 1 -it --rm -v ./data:/usr/src/app/data -v ./tsne/results:/usr/src/app/tsne/results -v ./pca/results:/usr/src/app/pca/results ./custom_clustering_utils/results:/usr/src/app/custom_clustering_utils/results larry:csiro_workstation
# curl "https://drive.usercontent.google.com/download?id={FILE_ID}&confirm=xxx" -o FILENAME.zip
