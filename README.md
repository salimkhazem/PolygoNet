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
```bash
python3 data/generate_sdp.py INPUT_PATH OUTPUT_PATH
