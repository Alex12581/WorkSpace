# -*- coding: utf-8 -*-

'''
this version works on the windows
'''

import requests
import re
import time
import os
from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFont
import subprocess

time = time.localtime()
imgpath = 'C:/Users/23755/OneDrive/图片/Wallpapers/bing_%s_%s_%s.jpg' % (time.tm_year, time.tm_mon, time.tm_mday)

# url = 'https://www.bing.com'
url = 'https://cn.bing.com'
headers = {'user-agent': 'Mozilla/5.0'}

logpath = 'E:/autoUpateWallpapers.log'
log = open(logpath, 'a', encoding='utf-8')

r = requests.get(url, headers=headers)
if r.status_code == 200:
    imgurl = url + '/' + re.search(r'background-image:url\(.*.jpg', r.text).group()[22:]  # 正则匹配图片链接
    # print(imgurl)
    # with open("test.txt", 'w', encoding="utf-8") as f:
        # f.write(r.text)
    imgname = BeautifulSoup(r.text, 'lxml').find('a', id='sh_cp').get('title')  # 获得图片标题
    for i in range(30):
        img = requests.get(imgurl)  # 下载图片
        if img.status_code == 200:
            with open(imgpath, 'wb') as fp:  # 保存图片
                fp.write(img.content)
            # 将图片标题加入图片右下角并重新保存
            with Image.open(imgpath) as img:
                x, y = img.size
                draw = ImageDraw.Draw(img)
                font = ImageFont.truetype('C:/Windows/Fonts/msyhbd.ttc', 20)
                draw.text((x-len(imgname)*13, y-30), imgname, (255,255,255), font=font)
                img.save(imgpath)
            subprocess.call("powershell.exe . E:/Set-Wallpaper; Set-Wallpaper -Path %s" % imgpath, True)
            print('%s.%s.%s %s:%s:%s更换了一次壁纸' % (time.tm_year, time.tm_mon, time.tm_mday, time.tm_hour, time.tm_min, time.tm_sec), file=log)
            break
    else:  #顺利循环完 30 次没有 break，说明下载失败
        print('Error happens when download image. (already try 30 times)', file=log)
else:
    print('Error happens when get main page.', file=log)
fp.close()
