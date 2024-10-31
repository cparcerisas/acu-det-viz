# Example scripts for visualizing Acoustic Detections using spotlight 

## Usage 
You will need to create a folder (the db path) with the following subfolders:

```bash 
path_to_folder
├── deployments
├── detections
│   ├── snippets
│   └── spectrograms
└── instruments (currently not necessary but will be the future)
```

To run the interface: 
0. Prepare the data. For this you need to create one json file per each deployment you want to process (one folder with recordings). This json file should contain, minimum, the fields "DEPLOYMENT_ID" and "FOLDER_PATH".
In the future it will also be necessary to have the "instrument" field. See config_example.json for an example.
All the config files need to have the same fields. 
1. Run add_files_deployment.py. This will read all the json configs from the folder deployments and create a csv file with all the raw wav files from all the deployments (included in the specified folders).
Right now RTSys instruments are used to obtain the datetime, this will change in the future. 
2. Run add_detections.py. Enter the path to the Raven txt file and its corresponding deployment id when asked for it. 
3. Run run_interface.py. This will start the spotlight interactive explorer 

