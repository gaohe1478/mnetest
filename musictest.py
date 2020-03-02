# -*- coding: utf-8 -*-
# @Author: Four Leaf Clover
# @Date:   2017-11-11 09:27:37
# @Last Modified by:   Administrator
# @Last Modified time: 2017-11-11 12:05:45

import pygame
import sys
from pygame.locals import *

# pygame和pygame.mixer初始化
pygame.init()
pygame.mixer.init()

# 设置及播放背景音乐
pygame.mixer.music.load("D:\CloudMusic\玖壹壹 - 下辈子.mp3")
pygame.mixer.music.set_volume(0.2)  # 设置音量
pygame.mixer.music.play()  # 播放音乐

# 设置两种不同的音效
left_sound = pygame.mixer.Sound("D:\CloudMusic\Juliet - 消えない花火.wav")
left_sound.set_volume(0.2)

right_sound = pygame.mixer.Sound("D:\CloudMusic\买辣椒也用券 - 起风了（Cover 高橋優）.wav")
right_sound.set_volume(0.2)

# 设置界面窗口
bg_size = width, height = 300, 200
bg_rgb = (255, 255, 255)
screen = pygame.display.set_mode(bg_size)
pygame.display.set_caption("Music - Four Leaf Clover")

# 创建设置帧率对象
clock = pygame.time.Clock()

# 创建播放和暂停图片surface对象
play_image = pygame.image.load("1.png").convert_alpha()
pause_image = pygame.image.load("2.png").convert_alpha()

# 获取播放和暂停矩形框
pause_rect = pause_image.get_rect()
pause_rect.left, pause_rect.top = (width - pause_rect.width) // 2, (height - pause_rect.height) // 2

# 定义播放标志位
pause = False

while True:
    # 查找队列事件
    for event in pygame.event.get():
        # 查找点击关闭窗口事件
        if event.type == QUIT:
            sys.exit()

        # 查找鼠标左右击事件
        if event.type == MOUSEBUTTONDOWN:
            # 检测鼠标左击是否按下
            if event.button == 1:
                left_sound.play()

            # 检测鼠标右击是否按下
            if event.button == 3:
                right_sound.play()

        # 检测键是否按下
        if event.type == KEYDOWN:
            # 检测是否为空格键按下
            if event.key == K_SPACE:
                pause = not pause

    # 填充界面背景
    screen.fill(bg_rgb)

    # 空格控制播放和暂停，并显示相应的图片
    if pause:
        pygame.mixer.music.pause()
        screen.blit(pause_image, pause_rect)
    else:
        pygame.mixer.music.unpause()
        screen.blit(play_image, pause_rect)

    # 刷新缓冲区图像
    pygame.display.flip()

    # 控制帧率为30帧
    clock.tick(30)