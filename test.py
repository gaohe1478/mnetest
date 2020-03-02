import mne
import numpy as np
import matplotlib.pyplot as plt

"""
生成一个大小为5x1000的二维随机数据
其中5代表5个通道，1000代表times
"""
data = np.random.randn(5, 1000)

"""
创建info结构,
内容包括：通道名称和通道类型
设置采样频率为:sfreq=100
"""
info = mne.create_info(
    ch_names=['MEG1', 'MEG2', 'EEG1', 'EEG2', 'EOG'],
    ch_types=['grad', 'grad', 'eeg', 'eeg', 'eog'],
    sfreq=100
)
"""
利用mne.io.RawArray类创建Raw对象
"""
custom_raw = mne.io.RawArray(data, info)
print(custom_raw)
"""
对图形进行缩放

对于实际的EEG / MEG数据，应使用不同的比例因子。
对通道eeg、grad，eog的数据进行2倍缩小
"""
scalings = {'eeg': 2, 'grad': 2,'eog':2}
custom_raw.plot(n_channels=5,
                scalings=scalings,
                title='Data from arrays',
         show=True, block=True)

plt.show()
