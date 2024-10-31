import pathlib
import pandas as pd
import dataset

db_path = pathlib.Path(input('Where is the folder path?'))

txt_path = pathlib.Path(input('Where is the txt file to add? '))
pointer_to_deployment = int(input('What is the deployment id? (in the future, select from drop list?)'))

all_files = pd.read_csv(db_path.joinpath('deployments', 'all_files.csv'), index_col=0)

files_deployment = all_files.loc[all_files.deployment_id == pointer_to_deployment]

config = {
    'wavs_folder': pathlib.Path(files_deployment.iloc[0].wav).parent,
    'wavs_files_df': files_deployment,
    'spectrograms_folder': db_path.joinpath('detections', 'spectrograms'),
    'snippets_folder': db_path.joinpath('detections', 'snippets'),
    'annotations_file': txt_path
}
ds = dataset.DetectionsDataset(config)

# raise error if wav file names don't match between txt annotations and deployment
# TODO

# compute snippets, spectrograms and embeddings
detections = ds.process()

# Read the other detections if they already exist
all_detections_db_path = db_path.joinpath('detections', 'all_detections.pkl')
if all_detections_db_path.exists():
    all_detections = pd.read_pickle(all_detections_db_path)
    all_detections = pd.concat([all_detections, detections])
else:
    all_detections = detections

all_detections.to_pickle(all_detections_db_path)

