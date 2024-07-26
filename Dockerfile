FROM python:3.10.1-buster

## DO NOT EDIT these 3 lines.
RUN mkdir /challenge
COPY ./ /challenge
WORKDIR /challenge

## Install your dependencies here using apt install, etc.

## Include the following line if you have a requirements.txt file.
RUN pip3 install -r requirements.txt --retries=5
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

## Download private denoise model from Google Drive
## RUN gdown --id 1RdiY5zeuj8ipItJxCbieXPFPlEDhwf9S -O ResNet_C_torch_all.model
## RUN gdown --id 1iA7wzbN7XJ7wJnWtPRZ6l_Kep3EcdZ8- -O UNet_torch_all.model
