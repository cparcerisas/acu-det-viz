import json
import pathlib
import pandas as pd
import pyhydrophone as pyhy
import soundfile as sf

db_path = pathlib.Path(input('Where is the folder path?'))


# This needs to be changed to it's read from the instrument metadata!
rtsys_name = 'RTSys'
rtsys_model = 'RESEA320'
rtsys_sensitivity = -180
rtsys_preamp_gain = 0
rtsys_Vpp = 2.0
serial_number = 3
rtsys = pyhy.RTSys(name=rtsys_name, model=rtsys_model, serial_number=serial_number, sensitivity=rtsys_sensitivity,
                   preamp_gain=rtsys_preamp_gain, Vpp=rtsys_Vpp, mode='lowpower', channel='A')

# Read all the json files inside the deployments folder
deployments_folder = db_path.joinpath('deployments')
columns = ['wav', 'wav_name', 'datetime', 'duration']
all_files_df = None
for deployment_path in deployments_folder.glob('*.json'):
    f = open(deployment_path)
    config = json.load(f)
    extra_columns = list(set(config.keys()) - set(columns))
    if all_files_df is None:
        all_files_df = pd.DataFrame(columns=columns + extra_columns)

    deployment_folder = pathlib.Path(config['FOLDER_PATH'])
    wav_list = []
    for wav_f in deployment_folder.glob('*.wav'):
        wav_datetime = rtsys.get_name_datetime(wav_f.name)
        wav_info = sf.SoundFile(wav_f)
        duration = wav_info.frames / wav_info.samplerate
        extra_values = []
        for c in extra_columns:
            extra_values.append(config[c])
        all_files_df.loc[len(all_files_df)] = [wav_f, wav_f.name, wav_datetime, duration] + extra_values

# Create a csv with all the wav files
all_files_df.to_csv(db_path.joinpath('deployments', 'all_files.csv'))