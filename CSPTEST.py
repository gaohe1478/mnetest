import numpy as np
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import ShuffleSplit, cross_val_score
from mne import Epochs, pick_types, events_from_annotations
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci
from mne.decoding import CSP
from pathlib import Path

###############################################################################
tmin, tmax = -1., 4.  # 设置参数，记录点的前1秒后4秒用于生成epoch数据
event_id = dict(hands=2, feet=3)  # 设置事件的映射关系
subject = 1
runs = [4,8,12]
#p = ("D:/eegdata/a20121020.edf")
# 获取想要读取的文件名称，这个应该是没有会默认下载的数据
#edfpath = eegbci.data_path(p)
raw_fnames = eegbci.load_data(subject, runs)
# 将3个文件的数据进行拼接
raw = concatenate_raws([read_raw_edf(f, preload=True) for f in raw_fnames])
# 去掉通道名称后面的（.），不知道为什么默认情况下raw.info['ch_names']中的通道名后面有的点
raw.rename_channels(lambda x: x.strip('.'))
# 对原始数据进行FIR带通滤波
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')  #
# 从annotation中获取事件信息
events, _ = events_from_annotations(raw, event_id=dict(T1=2, T2=3))
# 剔除坏道，提取其中有效的EEG数据
picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False, exclude='bads')
# 根据事件生成对应的Epochs数据
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks, baseline=None, preload=True)
# 截取其中的1秒到2秒之间的数据，也就是提示音后1秒到2秒之间的数据（这个在后面滑动窗口验证的时候有用）
epochs_train = epochs.copy().crop(tmin=1., tmax=2.)
# 将events转换为labels,event为2,3经过计算后也就是0,1
labels = epochs.events[:, -1] - 2

###########################第二部分，特征提取和分类####################################################
scores = []
# 获取epochs的所有数据，主要用于后面的滑动窗口验证
epochs_data = epochs.get_data()
# 获取训练数据
epochs_data_train = epochs_train.get_data()
# 设置交叉验证模型的参数
cv = ShuffleSplit(10, test_size=0.2, random_state=42)
# 根据设计的交叉验证参数,分配相关的训练集和测试集数据
cv_split = cv.split(epochs_data_train)
# 创建线性分类器
lda = LinearDiscriminantAnalysis()
# 创建CSP提取特征，这里使用4个分量的CSP
csp = CSP(n_components=4, reg=None, log=False, norm_trace=False)
# 创建机器学习的Pipeline,也就是分类模型，使用这种方式可以把特征提取和分类统一整合到了clf中
clf = Pipeline([('CSP', csp), ('LDA', lda)])
# 获取交叉验证模型的得分
scores = cross_val_score(clf, epochs_data_train, labels, cv=cv, n_jobs=1)
# 输出结果，准确率和不同样本的占比
class_balance = np.mean(labels == labels[0])
class_balance = max(class_balance, 1. - class_balance)
print("Classification accuracy: %f / Chance level: %f" % (np.mean(scores), class_balance))
# csp提取特征,用于绘制CSP不同分量的模式图（地形图）
# 如果没有这一步csp.plot_patterns将不会执行
csp.fit_transform(epochs_data, labels)
# lay文件的存放路径，这个文件不是计算生成的，是mne库提供的点击分布描述文件在安装路径下（根据个人安装路径查找）：
# D:\ProgramData\Anaconda3\Lib\site-packages\mne\channels\data\layouts\EEG1005.lay
layout = read_layout('EEG1005')
csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg', units='Patterns (AU)', size=1.5)

#########################验证算法的性能###########################################
# 获取数据的采样频率
sfreq = raw.info['sfreq']
# 设置滑动窗口的长度，也就是数据窗口的长度
w_length = int(sfreq * 0.5)
# 设置滑动步长，每次滑动的数据间隔
w_step = int(sfreq * 0.1)
# 每次滑动窗口的起始点
w_start = np.arange(0, epochs_data.shape[2] - w_length, w_step)
# 得分列表用于保存模型得分
scores_windows = []
# 交叉验证计算模型的性能
for train_idx, test_idx in cv_split:
    # 获取测试集和训练集数据
    y_train, y_test = labels[train_idx], labels[test_idx]
    # 设置csp模型的参数，提取相关特征，用于后面的lda分类
    X_train = csp.fit_transform(epochs_data_train[train_idx], y_train)
    # 拟合lda模型
    lda.fit(X_train, y_train)
    # 用于记录本次交叉验证的得分
    score_this_window = []
    for n in w_start:
        # csp提取测试数据相关特征
        X_test = csp.transform(epochs_data[test_idx][:, :, n:(n + w_length)])
        # 获取测试数据得分
        score_this_window.append(lda.score(X_test, y_test))
    # 添加到总得分列表
    scores_windows.append(score_this_window)

# 设置绘图的时间轴，时间轴上的标志点为窗口的中间位置
w_times = (w_start + w_length / 2.) / sfreq + epochs.tmin
# 绘制模型分类结果的性能图（得分的均值）
plt.figure()
plt.plot(w_times, np.mean(scores_windows, 0), label='Score')
plt.axvline(0, linestyle='--', color='k', label='Onset')
plt.axhline(0.5, linestyle='-', color='k', label='Chance')
plt.xlabel('time (s)')
plt.ylabel('classification accuracy')
plt.title('Classification score over time (w_length = {0}s)'.format(w_length / sfreq))
plt.legend(loc='lower right')
plt.show()