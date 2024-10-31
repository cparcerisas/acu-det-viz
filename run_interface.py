from renumics import spotlight
import pandas as pd
import pathlib

db_path = pathlib.Path(input('Where is the folder path?'))

all_detections_path = db_path.joinpath('detections', 'all_detections.pkl')
all_detections = pd.read_pickle(all_detections_path)

embeddings = all_detections.iloc[:, 0:512]
detections_metadata = all_detections.iloc[:, 512:]
detections_metadata['embeddings'] = embeddings.values.tolist()
detections_metadata['image_path'] = str(db_path.joinpath('detections', 'spectrograms')) + '\\' + detections_metadata.deployment_id.astype(str) + '_' + detections_metadata.index.astype(str) + '.png'
detections_metadata['audio_path'] = str(db_path.joinpath('detections', 'snippets')) + '\\' + detections_metadata.deployment_id.astype(str) + '_' + detections_metadata.index.astype(str) + '.wav'
detections_metadata = detections_metadata.drop(columns=['wav'])
spotlight.show(detections_metadata, port=64607)
