# -*- coding: utf-8 -*-

'''
this version works on the windows
'''

from requests import get
from re import search
from time import localtime
from os import environ, mkdir 
from os.path import exists
from bs4 import BeautifulSoup
from PIL.Image import open as imgopen
from PIL.ImageDraw import Draw
from PIL.ImageFont import truetype
import subprocess

time = localtime()
root = 'C:'+environ['HOMEPATH']+'\\BingWallpapers\\'
if not exists(root):
    mkdir(root)
imgpath = root+'bing_%s_%s_%s.jpg' % (time.tm_year, time.tm_mon, time.tm_mday)

url = 'https://cn.bing.com'
headers = {'user-agent': 'Mozilla/5.0'}

r = get(url, headers=headers)
if r.status_code == 200:
    imgurl = url + '/' + search(r'background-image:url\(.*.jpg', r.text).group()[22:]  # 正则匹配图片链接
    imgname = BeautifulSoup(r.text, 'lxml').find('a', id='sh_cp').get('title')  # 获得图片标题
    for i in range(30):
        img = get(imgurl)  # 下载图片
        if img.status_code == 200:
            with open(imgpath, 'wb') as fp:  # 保存图片
                fp.write(img.content)
            # 将图片标题加入图片右下角并重新保存
            with imgopen(imgpath) as img:
                x, y = img.size
                draw = Draw(img)
                font = truetype('C:/Windows/Fonts/msyhbd.ttc', 20)
                draw.text((x-len(imgname)*13, y-30), imgname, (255,255,255), font=font)
                img.save(imgpath)
            subprocess.call("powershell.exe . ./Set-Wallpaper; Set-Wallpaper -Path %s" % imgpath, False)
            break
    else:  #顺利循环完 30 次没有 break，说明下载失败
        pass
else:
    pass
