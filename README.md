## Dependencies 
For the different parts you can install required dependecies by running the following command: 

### For MATC
  #### Install in local 
```bash 
  bash install_MATC.sh
````
  #### Use Docker 
```bash
  docker build -t matc .
```
### For Deep Learning pipeline 
  ```bash
  pip install -r requirements.txt 
  ````
## Usage 
### Contours extraction and SDP files generation
`INPUT_PATH` is the path to the folder containing the images. `OUTPUT_PATH` is the path to the folder where the SDP files will be saved. 
```bash
python3 data/generate_sdp.py INPUT_PATH OUTPUT_PATH
