import pandas as pd
from glob import glob

# # --------------------------------------------------------------
# # Read single CSV file
# # --------------------------------------------------------------
# single_file_acc = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Accelerometer_12.500Hz_1.4.4.csv") 

# single_file_gyr = pd.read_csv("../../data/raw/MetaMotion/A-bench-heavy2-rpe8_MetaWear_2019-01-11T16.10.08.270_C42732BE255C_Gyroscope_25.000Hz_1.4.4.csv")

# # --------------------------------------------------------------
# # List all data in data/raw/MetaMotion
# # --------------------------------------------------------------
# files = glob("../../data/raw/MetaMotion/*.csv")

# # --------------------------------------------------------------
# # Extract features from filename
# # --------------------------------------------------------------
# data_path = "../../data/raw/MetaMotion\\"
# f = files[1]

# participant = f.split('-')[0].replace(data_path,'')
# label = f.split('-')[1]
# category = f.split('-')[2].rstrip('123').rstrip('_MetaWear_2019')

# df = pd.read_csv(f)

# df['Participant'] = participant
# df['Label'] = label
# df['Category'] = category
# # --------------------------------------------------------------
# # Read all files
# # --------------------------------------------------------------

# acc_df = pd.DataFrame()
# gyr_df = pd.DataFrame()

# acc_set = 1
# gyr_set = 1

# for f in files:
#     participant = f.split('-')[0].replace(data_path,'')
#     label = f.split('-')[1]
#     category = f.split('-')[2].rstrip('123').rstrip('_MetaWear_2019')

#     df = pd.read_csv(f)

#     df['Participant'] = participant
#     df['Label'] = label
#     df['Category'] = category
    
# if 'Accelerometer' in f:
#      df['Set'] = acc_set
#      acc_set += 1
#      acc_df = pd.concat([acc_df,df])
#  if 'Gyroscope'in f:
#      df['Set'] = gyr_set
#      gyr_set += 1
#      gyr_df = pd.concat([gyr_df,df])

# # --------------------------------------------------------------
# # Working with datetimes
# # --------------------------------------------------------------
# acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
# gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

# drop_col = ['epoch (ms)', 'time (01:00)', 'elapsed (s)']
# acc_df = acc_df.drop(columns = drop_col)
# gyr_df = gyr_df.drop(columns = drop_col)

# --------------------------------------------------------------
# Turn into function
# --------------------------------------------------------------

files = glob("../../data/raw/MetaMotion/*.csv")

def read_csv_files(files):
    
    acc_df = pd.DataFrame()
    gyr_df = pd.DataFrame()

    data_path = "../../data/raw/MetaMotion\\"
    
    acc_set = 1
    gyr_set = 1

    for f in files:
        participant = f.split('-')[0].replace(data_path,'')
        label = f.split('-')[1]
        category = f.split('-')[2].rstrip('123').rstrip('_MetaWear_2019')

        df = pd.read_csv(f)

        df['Participant'] = participant
        df['Label'] = label
        df['Category'] = category
        
        if 'Accelerometer' in f:
            df['Set'] = acc_set
            acc_set += 1
            acc_df = pd.concat([acc_df,df])
        if 'Gyroscope'in f:
            df['Set'] = gyr_set
            gyr_set += 1
            gyr_df = pd.concat([gyr_df,df]) 
    
    
    acc_df.index = pd.to_datetime(acc_df['epoch (ms)'], unit='ms')
    gyr_df.index = pd.to_datetime(gyr_df['epoch (ms)'], unit='ms')

    drop_col = ['epoch (ms)', 'time (01:00)', 'elapsed (s)']
    acc_df = acc_df.drop(columns = drop_col)
    gyr_df = gyr_df.drop(columns = drop_col)

    return acc_df,gyr_df

acc_df,gyr_df = read_csv_files(files=files)
# --------------------------------------------------------------
# Merging datasets
# --------------------------------------------------------------
merged_data = pd.concat([acc_df.iloc[:,:3],gyr_df], axis=1)
merged_data.columns = [
    'acc_x', 
    'acc_y',
    'acc_z',
    'gyr_x',
    'gyr_y',
    'gyr_z',
    'Participant',
    'Label',
    'Category',
    'Set'
]

# --------------------------------------------------------------
# Resample data (frequency conversion)
# --------------------------------------------------------------

# Accelerometer:    12.500HZ
# Gyroscope:        25.000Hz

sampling = {
    'acc_x':'mean', 
    'acc_y':'mean',
    'acc_z':'mean',
    'gyr_x':'mean',
    'gyr_y':'mean',
    'gyr_z':'mean',
    'Participant':'last',
    'Label':'last',
    'Category':'last',
    'Set':'last'
}

days = [g for n, g in merged_data.groupby(pd.Grouper(freq='D'))]
data_resampled = pd.concat([df.resample(rule='200ms').apply(sampling).dropna() for df in days])

data_resampled['Set'] = data_resampled['Set'].astype('int')
# --------------------------------------------------------------
# Export dataset
# --------------------------------------------------------------
data_resampled.to_pickle('../../data/interim/01_data_processed')
