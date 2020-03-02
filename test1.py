import mne
path = "D:\\eegdata\\left-right _hand\\"
file_name = "B0101T.gdf"
raw_gdf = mne.io.read_raw_gdf(path+file_name)
#print(raw_gdf.annotations.onset)
print(type(raw_gdf))
print(raw_gdf.annotations)