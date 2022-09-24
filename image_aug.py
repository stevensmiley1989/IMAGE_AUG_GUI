'''
Img Aug
========
Created by Steven Smiley 5/28/2022

image_aug.py is a GUI for creating custom augmented datasets using imgaug. 

It is written in Python and uses Tkinter for its graphical interface.


Installation
------------------

Ubuntu Linux
~~~~~~~

Python 3 + Tkinter + Darknet YOLO

.. code:: shell
    cd ~/
    cd IMAGE_AUG_GUI
    sudo pip3 install -r requirements.txt
    python3 image_aug.py
~~~~~~~
'''
import codecs
import numpy as np

import functools
import time
import PIL
from PIL import Image, ImageTk
Image.MAX_IMAGE_PIXELS = None #edit sjs 5/28/2022
from PIL import ImageDraw
from PIL import ImageFont
import tkinter as tk
from tkinter import Toplevel, ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
from tkinter.tix import Balloon
from skimage.metrics import structural_similarity as ssim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap 
from pprint import pprint
import cv2
import sys
from sys import platform as _platform
import os
import cv2
import imgaug as ia
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from xml.etree import ElementTree
from xml.etree.ElementTree import Element, SubElement
from lxml import etree
import codecs
from tqdm import tqdm
import cv2
import pandas as pd
import shutil
from functools import partial
from threading import Thread
import traceback
from imgaug import parameters as iap
XML_EXT = '.xml'
DEFAULT_ENCODING = 'utf-8'
ENCODE_METHOD = DEFAULT_ENCODING
def get_default_settings(SAVED_SETTINGS='SAVED_SETTINGS'):
    global DEFAULT_SETTINGS
    try:
        #from libs import SAVED_SETTINGS as DEFAULT_SETTINGS
        exec('from libs import {} as DEFAULT_SETTINGS'.format(SAVED_SETTINGS),globals())
        if os.path.exists(DEFAULT_SETTINGS.path_JPEGImages):
            pass
        else:
            from libs import DEFAULT_SETTINGS
    except:
        print('Using Original DEFAULT_SETTINGS')
        from libs import DEFAULT_SETTINGS 
class main_entry:
    global SAVED_SETTINGS_PATH
    def __init__(self,root_tk):
        self.root=root_tk
        self.root.bind('<Escape>',self.close)
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appI.png")))
        self.icon_breakup=ImageTk.PhotoImage(Image.open('resources/icons/breakup.png'))
        self.icon_folder=ImageTk.PhotoImage(Image.open("resources/icons/open_folder_search.png"))
        self.icon_single_file=ImageTk.PhotoImage(Image.open('resources/icons/single_file.png'))
        self.icon_load=ImageTk.PhotoImage(Image.open('resources/icons/load.png'))
        self.icon_create=ImageTk.PhotoImage(Image.open('resources/icons/create.png'))
        self.icon_analyze=ImageTk.PhotoImage(Image.open('resources/icons/analyze.png'))
        self.icon_move=ImageTk.PhotoImage(Image.open('resources/icons/move.png'))
        self.icon_labelImg=ImageTk.PhotoImage(Image.open('resources/icons/labelImg.png'))
        self.icon_map=ImageTk.PhotoImage(Image.open('resources/icons/map.png'))
        self.icon_merge=ImageTk.PhotoImage(Image.open('resources/icons/merge.png'))
        self.icon_clear_fix=ImageTk.PhotoImage(Image.open('resources/icons/clear.png'))
        self.icon_clear_checked=ImageTk.PhotoImage(Image.open('resources/icons/clear_checked.png'))
        self.icon_save_settings=ImageTk.PhotoImage(Image.open('resources/icons/save.png'))
        self.icon_yolo_objects=ImageTk.PhotoImage(Image.open('resources/icons/yolo_objects.png'))
        self.icon_divide=ImageTk.PhotoImage(Image.open('resources/icons/divide.png'))
        self.icon_scripts=ImageTk.PhotoImage(Image.open('resources/icons/scripts.png'))
        self.icon_config=ImageTk.PhotoImage(Image.open('resources/icons/config.png'))
        self.icon_open=ImageTk.PhotoImage(Image.open('resources/icons/open_folder.png'))
        self.icon_train=ImageTk.PhotoImage(Image.open('resources/icons/train.png'))
        self.icon_test_mp4=ImageTk.PhotoImage(Image.open('resources/icons/test_mp4.png'))
        self.icon_test_images=ImageTk.PhotoImage(Image.open('resources/icons/test_images.png'))
        self.icon_test=ImageTk.PhotoImage(Image.open('resources/icons/test.png'))
        self.style4=ttk.Style()
        self.style4.configure('Normal.TCheckbutton',
                             background='green',
                             foreground='black')
        self.root_H=int(self.root.winfo_screenheight()*0.95)
        self.root_W=int(self.root.winfo_screenwidth()*0.95)
        self.root.geometry(str(self.root_W)+'x'+str(self.root_H))
        self.root.title("IMG AUG GUI")
        self.root_bg='black'
        self.root_fg='lime'
        self.canvas_columnspan=50
        self.canvas_rowspan=50
        self.root_background_img=r"misc/gradient_green.jpg"
        self.get_update_background_img()
        self.root.configure(bg=self.root_bg)
        self.dropdown=None
        self.CWD=os.getcwd()
        self.df_settings=pd.DataFrame(columns=['files','Annotations'])
        self.SETTINGS_FILE_LIST=[w.split('.py')[0] for w in os.listdir('libs') if w.find('SETTINGS')!=-1 and w[0]!='.'] 
        self.files_keep=[]
        i=0
        for file in self.SETTINGS_FILE_LIST:
            file=file+'.py'
            if file!="DEFAULT_SETTINGS.py":
                found=False
                f=open(os.path.join('libs',file),'r')
                f_read=f.readlines()
                f.close()
                for line in f_read:
                    if line.find('path_Annotations')!=-1:
                        self.files_keep.append(file.split('.py')[0])
                        self.df_settings.at[i,'files']=file.split('.py')[0]
                        self.df_settings.at[i,'Annotations']=line.split('=')[-1].replace("'",'"').split('"')[1].split('Annotations')[0].split('/')[-2]
                        found=True
                if found==True:
                    i+=1
        self.df_settings=self.df_settings.fillna(0)
        self.files_keep.append('DEFAULT_SETTINGS')
        print(self.df_settings)
        self.checkd_buttons={}
        self.checkd_vars={}
        self.checkd_label=tk.Label(self.root,text='Dataset',bg=self.root_bg,fg=self.root_fg,font=('Arial 14 underline'))
        self.checkd_label.grid(row=1,column=2,sticky='nw')
        for i,label in enumerate(list(self.df_settings['Annotations'].unique())):
            self.checkd_vars[label]=tk.IntVar()
            self.checkd_vars[label].set(1)
            self.checkd_buttons[file]=ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text=label,variable=self.checkd_vars[label], command=self.update_checks,onvalue=1, offvalue=0)
            self.checkd_buttons[file].grid(row=i+1,column=2,sticky='sw')         


        self.SETTINGS_FILE_LIST=self.files_keep
        self.df_comb=pd.DataFrame(columns=['times','items'])
        self.df_comb['times']=[os.path.getmtime(os.path.join('libs/',w+'.py')) for w in self.SETTINGS_FILE_LIST]
        self.df_comb['items']=[w for w in self.SETTINGS_FILE_LIST]
        self.df_comb=self.df_comb.sort_values(by='times',ascending=True).reset_index().drop('index',axis=1)
        self.SETTINGS_FILE_LIST=list(self.df_comb['items'])
        self.USER=""
        self.USER_SELECTION=tk.StringVar()
        self.dropdown_menu()
        self.submit_label=Button(self.root,text='Submit',command=self.submit,bg=self.root_fg,fg=self.root_bg,font=('Arial',12))
        self.submit_label.grid(row=1,column=5,sticky='se')


    def update_checks(self):
        checked_datasets=[]
        for dataset,var in self.checkd_vars.items():
            if var.get()==1:
                checked_datasets.append(dataset)
        df_temp=self.df_settings[(self.df_settings['Annotations'].isin(checked_datasets))].copy()
        self.files_keep=list(df_temp['files'])
        self.files_keep.append('DEFAULT_SETTINGS')
        self.dropdown_menu()

            
    def dropdown_menu(self):
        if self.dropdown!=None:
            self.dropdown_label.destroy()
            self.dropdown.destroy()
        self.SETTINGS_FILE_LIST=self.files_keep
        self.df_comb=pd.DataFrame(columns=['times','items'])
        self.df_comb['times']=[os.path.getmtime(os.path.join('libs/',w+'.py')) for w in self.SETTINGS_FILE_LIST]
        self.df_comb['items']=[w for w in self.SETTINGS_FILE_LIST]
        self.df_comb=self.df_comb.sort_values(by='items',ascending=True).reset_index().drop('index',axis=1)
        self.SETTINGS_FILE_LIST=list(self.df_comb['items'])
   
        self.USER_SELECTION=tk.StringVar()
        if self.USER in self.SETTINGS_FILE_LIST:
            self.USER_SELECTION.set(self.USER)
        else:
            self.USER_SELECTION.set(self.SETTINGS_FILE_LIST[0])
        self.dropdown=tk.OptionMenu(self.root,self.USER_SELECTION,*self.SETTINGS_FILE_LIST)
        self.dropdown.grid(row=1,column=9,sticky='sw')
        
        self.dropdown_label=Button(self.root,image=self.icon_single_file,command=self.run_cmd_libs,bg=self.root_bg,fg=self.root_fg,font=('Arial',12))
        self.dropdown_label.grid(row=1,column=8,sticky='sw')
       
    def run_cmd_libs(self):
        cmd_i=open_cmd+" libs/{}.py".format(self.USER_SELECTION.get())
        os.system(cmd_i)

    def submit(self):
        global SAVED_SETTINGS_PATH
        self.USER=self.USER_SELECTION.get()
        self.USER=self.USER.strip()
        get_default_settings(self.USER)
        SAVED_SETTINGS_PATH=os.path.join('libs/{}'.format(self.USER))
        self.close()

    def get_update_background_img(self):
        self.image=Image.open(self.root_background_img)
        self.image=self.image.resize((self.root_W,self.root_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.root,width=self.root_W,height=self.root_H)
        self.canvas.grid(row=0,column=0,columnspan=self.canvas_columnspan,rowspan=self.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')

    def close(self):
        self.root.destroy()

class PascalVocReader:
    def __init__(self, file_path,EXT='.jpg',gt_path=None):
        # shapes type:
        # [labbel, [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]]
        self.shapes = []
        self.file_path = file_path
        self.verified = False
        self.gt_path=gt_path
        self.EXT=EXT
        try:
            self.parse_xml()
        except:
            pass

    def get_shapes(self):
        return self.shapes

    def add_shape(self, label, bnd_box):
        x_min = int(float(bnd_box.find('xmin').text))
        y_min = int(float(bnd_box.find('ymin').text))
        x_max = int(float(bnd_box.find('xmax').text))
        y_max = int(float(bnd_box.find('ymax').text))
        points = (x_min, y_min,x_max, y_max)
        self.shapes.append((label, points))

    def parse_xml(self):
        assert self.file_path.endswith(XML_EXT), "Unsupported file format"
        parser = etree.XMLParser(encoding=ENCODE_METHOD)
        xml_tree = ElementTree.parse(self.file_path, parser=parser).getroot()
        filename = xml_tree.find('filename').text
        self.img_path=xml_tree.find('path').text
        if self.img_path.find(self.EXT)==-1:
            self.img_path=os.path.join(self.img_path,filename)
        #print(self.img_path)
        if self.img_path.find(self.EXT)!=-1 and self.gt_path==None:
            if os.path.exists(self.img_path.replace(self.EXT,'.xml').replace('JPEGImages','Annotations')) and self.gt_path==None:
                self.gt_path=self.img_path.replace(self.EXT,'.xml').replace('JPEGImages','Annotations')
        for object_iter in xml_tree.findall('object'):
            bnd_box = object_iter.find("bndbox")
            label = object_iter.find('name').text
            self.add_shape(label, bnd_box)
        return True
def writePascalVOV(img_data,bboxes,path_jpg_old,path_anno_old,path_jpg_new,path_anno_new):
    height,width,depth=img_data.shape
    if height>width:
        print('height GREATER than width')
    #print(height,width,depth)
    folder='JPEGImages'
    filename=path_jpg_new.split('/')[-1]
    path=os.path.join(path_jpg_new,filename)
    cv2.imwrite(path_jpg_new,img_data)
    #shutil.copy(path_jpg_old,path_jpg_new)
    path=path_jpg_new
    database='Unknown'
    path_xml=path_anno_new
    f=open(path_xml,'w')
    f.writelines('<annotation>\n')
    f.writelines('\t <folder>{}</folder>\n'.format(folder))
    f.writelines('\t <filename>{}</filename>\n'.format(filename))
    f.writelines('\t <path>{}</path>\n'.format(path))
    f.writelines('\t<source>\n')
    f.writelines('\t\t<database>{}</database>\n'.format(database))
    f.writelines('\t</source>\n')
    f.writelines('\t<size>\n')
    f.writelines('\t\t<width>{}</width>\n'.format(width))
    f.writelines('\t\t<height>{}</height>\n'.format(height))
    f.writelines('\t\t<depth>{}</depth>\n'.format(depth))  
    f.writelines('\t</size>\n')
    f.writelines('\t<segmented>0</segmented>\n')
    count=0
    for box in bboxes:
        #print(box)
        xmin=int(box.x1)
        ymin=int(box.y1)
        xmax=int(box.x2)
        ymax=int(box.y2)
        xmin=abs(max(1,xmin))
        ymin=abs(max(1,ymin))
        xmax=abs(min(xmax,width-1))
        ymax=abs(min(ymax,height-1))
        if int(xmin)<0 or int(ymin)<0 or int(xmax)<0 or int(ymax)<0:
            #print('NEGATIVE')
            #print('xmin',xmin,'ymin',ymin,'xmax',xmax,'ymax',ymax)
            pass
        elif xmin>xmax or ymin>ymax:
            #print('wrong dimensions')
            #print('xmin',xmin,'ymin',ymin,'xmax',xmax,'ymax',ymax)
            pass
        else:
            count+=1
            name=box.label
            pose='Unspecified'
            truncated='0' 
            difficult='0'
            f.writelines('\t<object>\n')
            f.writelines('\t\t<name>{}</name>\n'.format(name))
            f.writelines('\t\t<pose>{}</pose>\n'.format(pose))       
            f.writelines('\t\t<truncated>{}</truncated>\n'.format(truncated))
            f.writelines('\t\t<difficult>{}</difficult>\n'.format(difficult))
            f.writelines('\t\t<bndbox>\n')
            f.writelines('\t\t\t<xmin>{}</xmin>\n'.format(xmin))
            f.writelines('\t\t\t<ymin>{}</ymin>\n'.format(ymin))
            f.writelines('\t\t\t<xmax>{}</xmax>\n'.format(xmax))
            f.writelines('\t\t\t<ymax>{}</ymax>\n'.format(ymax))
            f.writelines('\t\t</bndbox>\n')
            f.writelines('\t</object>\n')
    f.writelines('</annotation>\n')
    return count

if _platform=='darwin':
    import tkmacosx
    from tkmacosx import Button as Button
    open_cmd='open'
    matplotlib.use('MacOSX')
else:
    from tkinter import Button as Button
    if _platform.lower().find('linux')!=-1:
        open_cmd='xdg-open'
    else:
        open_cmd='start'
if os.path.exists('libs/open_cmd.py'):
    from libs import open_cmd
    open_cmd=open_cmd.open_cmd
root_tk=tk.Tk()
ROOT_H=int(root_tk.winfo_screenheight()*0.95)
ROOT_W=int(root_tk.winfo_screenwidth()*0.95)
print('ROOT_H',ROOT_H)
print('ROOT_W',ROOT_W)
def remove_directory(dir):
    #dir = 'path/to/dir'
    for files in os.listdir(dir):
        path = os.path.join(dir, files)
        try:
            shutil.rmtree(path)
        except OSError:
            os.remove(path)
    os.rmdir(dir)
class IMGAug_JPGS_ANNOS:
    def __init__(self,path_Annotations='None',path_JPEGImages='None'):
        self.root=root_tk
        self.root.bind('<Escape>',self.close)
        if path_JPEGImages=='None':
            self.path_JPEGImages=DEFAULT_SETTINGS.path_JPEGImages
        else:
            self.path_JPEGImages=path_JPEGImages
        if path_Annotations=='None':
            self.path_Annotations=DEFAULT_SETTINGS.path_Annotations
        else:
            self.path_Annotations=path_Annotations
        self.root_background_img=r'misc/gradient_green.jpg'
        self.root_W=ROOT_W
        self.root_H=ROOT_H
        self.root.wm_iconphoto(False,ImageTk.PhotoImage(Image.open("resources/icons/appI.png")))
        self.icon_breakup=ImageTk.PhotoImage(Image.open('resources/icons/breakup.png'))
        self.icon_folder=ImageTk.PhotoImage(Image.open("resources/icons/open_folder_search.png"))
        self.icon_single_file=ImageTk.PhotoImage(Image.open('resources/icons/single_file.png'))
        self.icon_load=ImageTk.PhotoImage(Image.open('resources/icons/load.png'))
        self.icon_create=ImageTk.PhotoImage(Image.open('resources/icons/create.png'))
        self.icon_analyze=ImageTk.PhotoImage(Image.open('resources/icons/analyze.png'))
        self.icon_move=ImageTk.PhotoImage(Image.open('resources/icons/move.png'))
        self.icon_labelImg=ImageTk.PhotoImage(Image.open('resources/icons/labelImg.png'))
        self.icon_map=ImageTk.PhotoImage(Image.open('resources/icons/map.png'))
        self.icon_merge=ImageTk.PhotoImage(Image.open('resources/icons/merge.png'))
        self.icon_clear_fix=ImageTk.PhotoImage(Image.open('resources/icons/clear.png'))
        self.icon_clear_checked=ImageTk.PhotoImage(Image.open('resources/icons/clear_checked.png'))
        self.icon_save_settings=ImageTk.PhotoImage(Image.open('resources/icons/save.png'))
        self.icon_yolo_objects=ImageTk.PhotoImage(Image.open('resources/icons/yolo_objects.png'))
        self.icon_divide=ImageTk.PhotoImage(Image.open('resources/icons/divide.png'))
        self.icon_scripts=ImageTk.PhotoImage(Image.open('resources/icons/scripts.png'))
        self.icon_config=ImageTk.PhotoImage(Image.open('resources/icons/config.png'))
        self.icon_open=ImageTk.PhotoImage(Image.open('resources/icons/open_folder.png'))
        self.icon_train=ImageTk.PhotoImage(Image.open('resources/icons/train.png'))
        self.icon_test_mp4=ImageTk.PhotoImage(Image.open('resources/icons/test_mp4.png'))
        self.icon_test_images=ImageTk.PhotoImage(Image.open('resources/icons/test_images.png'))
        self.icon_test=ImageTk.PhotoImage(Image.open('resources/icons/test.png'))

        self.path_labelImg=r'/media/steven/Elements/LabelImg/labelImg.py' #r'/Volumes/One Touch/labelImg-Custom/labelImg.py'
        self.jpeg_selected=False #after  user has opened from folder dialog the jpeg folder, this returns True
        self.anno_selected=False #after user has opened from folder dialog the annotation folder, this returns True
        self.root_H=int(self.root.winfo_screenheight()*0.95)
        self.root_W=int(self.root.winfo_screenwidth()*0.95)
        self.root.geometry(str(self.root_W)+'x'+str(self.root_H))
        self.root.title("IMG AUG GUI")
        self.predefined_classes='predefined_classes.txt'
        self.PYTHON_PATH=DEFAULT_SETTINGS.PYTHON_PATH #'python3'  
        self.PREFIX=DEFAULT_SETTINGS.PREFIX #'augmented'   
        self.TRAIN_SPLIT=DEFAULT_SETTINGS.TRAIN_SPLIT #'70.0'   
        self.root_bg=DEFAULT_SETTINGS.root_bg#'black'
        self.root_fg=DEFAULT_SETTINGS.root_fg#'lime'
        self.canvas_columnspan=DEFAULT_SETTINGS.canvas_columnspan
        self.canvas_rowspan=DEFAULT_SETTINGS.canvas_rowspan
        self.root_background_img=DEFAULT_SETTINGS.root_background_img #r"misc/gradient_blue.jpg"
        self.DEFAULT_ENCODING=DEFAULT_SETTINGS.DEFAULT_ENCODING
        self.XML_EXT=DEFAULT_SETTINGS.XML_EXT
        self.JPG_EXT=DEFAULT_SETTINGS.JPG_EXT
        self.COLOR=DEFAULT_SETTINGS.COLOR
        self.MAX_KEEP=DEFAULT_SETTINGS.MAX_KEEP #'2000'
        self.open_aug_jpeg_label_var=tk.StringVar()
        self.open_aug_jpeg_label_var.set('None')
        self.open_aug_anno_label_var=tk.StringVar()
        self.open_aug_anno_label_var.set('None')
        
        self.get_update_background_img()

        #self.root.config(menu=self.menubar)
        self.root.configure(bg=self.root_bg)
        self.drop_targets=None
        self.CWD=os.getcwd()
        self.not_checked_label=None
        self.checked_label_good=None
        self.checked_label_bad=None
        self.total_not_checked_label=None
        self.total_checked_label_good=None
        self.total_checked_label_bad=None
        
        self.open_anno_label_var=tk.StringVar()
        self.open_anno_label_var.set(self.path_Annotations)

        self.open_anno_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_Annotations,'Open Annotations Folder',self.open_anno_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_anno_button.grid(row=2,column=1,sticky='se')
        self.open_anno_note=tk.Label(self.root,text="1.a \n Annotations dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_anno_note.grid(row=3,column=1,sticky='ne')

        cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var.get())
        self.open_anno_label=Button(self.root,textvariable=self.open_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))

        self.open_anno_label.grid(row=2,column=2,columnspan=50,sticky='sw')

        self.open_jpeg_label_var=tk.StringVar()
        self.open_jpeg_label_var.set(self.path_JPEGImages)

        self.open_jpeg_button=Button(self.root,image=self.icon_folder,command=partial(self.select_folder,self.path_JPEGImages,'Open JPEGImages Folder',self.open_jpeg_label_var),bg=self.root_bg,fg=self.root_fg)
        self.open_jpeg_button.grid(row=4,column=1,sticky='se')
        self.open_jpeg_note=tk.Label(self.root,text="1.b \n JPEGImages dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_jpeg_note.grid(row=5,column=1,sticky='ne')

        cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var.get())
        self.open_jpeg_label=Button(self.root,textvariable=self.open_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))

        self.open_jpeg_label.grid(row=4,column=2,columnspan=50,sticky='sw')

        self.save_settings_button=Button(self.root,image=self.icon_save_settings,command=self.save_settings,bg=self.root_bg,fg=self.root_fg)
        self.save_settings_button.grid(row=1,column=4,sticky='se')
        self.save_settings_note=tk.Label(self.root,text='Save Settings',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
        self.save_settings_note.grid(row=2,column=4,sticky='ne')

        self.submit_button=Button(self.root,text='Submit',command=self.augment_my_imgs,bg=self.root_fg,fg=self.root_bg,font=("Arial 14 bold"))
        self.submit_button.grid(row=21,column=1,sticky='s')
        #self.submit_note=tk.Label(self.root,text="Submit",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        #self.submit_note.grid(row=18,column=1,sticky='ne')
        self.style3=ttk.Style()
        self.style3.configure('Normal.TRadiobutton',
                             background='black',
                             foreground='green')
        self.style4=ttk.Style()
        self.style4.configure('Normal.TCheckbutton',
                             background='green',
                             foreground='black')

        self.PREFIX_VAR=tk.StringVar()
        self.PREFIX_VAR.set(self.PREFIX)
        self.PREFIX_entry=tk.Entry(self.root,textvariable=self.PREFIX_VAR)
        self.PREFIX_entry.grid(row=13,column=1,sticky='sw')
        self.PREFIX_label=tk.Label(self.root,text='Annotation PREFIX',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.PREFIX_label.grid(row=14,column=1,sticky='nw')

        self.TRAIN_SPLIT_VAR=tk.StringVar()
        self.TRAIN_SPLIT_VAR.set(self.TRAIN_SPLIT)
        self.TRAIN_SPLIT_entry=tk.Entry(self.root,textvariable=self.TRAIN_SPLIT_VAR)
        self.TRAIN_SPLIT_entry.grid(row=15,column=1,sticky='sw')
        self.TRAIN_SPLIT_button=Button(self.root,image=self.icon_divide,command=self.load_my_imgs)
        self.TRAIN_SPLIT_button.grid(row=15,column=0,sticky='se')
        self.TRAIN_SPLIT_label=tk.Label(self.root,text='TRAIN SPLIT',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.TRAIN_SPLIT_label.grid(row=16,column=1,sticky='nw')

        self.MAX_KEEP_VAR=tk.StringVar()
        self.MAX_KEEP_VAR.set(self.MAX_KEEP)
        self.MAX_KEEP_entry=tk.Entry(self.root,textvariable=self.MAX_KEEP_VAR)
        self.MAX_KEEP_entry.grid(row=17,column=1,sticky='sw')
        self.MAX_KEEP_button=Button(self.root,image=self.icon_yolo_objects,command=self.load_my_imgs)
        self.MAX_KEEP_button.grid(row=17,column=0,sticky='se')
        self.MAX_KEEP_label=tk.Label(self.root,text='MAX # per CLASS',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.MAX_KEEP_label.grid(row=18,column=1,sticky='nw')

        self.var_sometimes=tk.IntVar()
        self.var_sometimes_INIT=DEFAULT_SETTINGS.var_sometimes_INIT
        self.var_sometimes.set(self.var_sometimes_INIT)#
        self.var_sometimes_frac=tk.StringVar()
        self.var_sometimes_frac_INIT=DEFAULT_SETTINGS.var_sometimes_frac_INIT
        self.var_sometimes_frac.set(self.var_sometimes_frac_INIT) #'0.5'

        self.var_Fliplr = tk.IntVar()
        self.var_Fliplr_INIT=DEFAULT_SETTINGS.var_Fliplr_INIT
        self.var_Fliplr.set(self.var_Fliplr_INIT) #1
        self.var_Fliplr_frac=tk.StringVar()
        self.var_Fliplr_frac_INIT=DEFAULT_SETTINGS.var_Fliplr_frac_INIT
        self.var_Fliplr_frac.set(self.var_Fliplr_frac_INIT)
        self.var_sometimes_Fliplr_frac=tk.StringVar()
        self.var_sometimes_Fliplr_frac_INIT=DEFAULT_SETTINGS.var_sometimes_Fliplr_frac_INIT
        self.var_sometimes_Fliplr_frac.set(self.var_sometimes_Fliplr_frac_INIT) #'0.5



        self.var_Flipud = tk.IntVar()
        self.var_Flipud_INIT=DEFAULT_SETTINGS.var_Flipud_INIT
        self.var_Flipud.set(self.var_Flipud_INIT)
        self.var_Flipud_frac=tk.StringVar()
        self.var_Flipud_frac_INIT=DEFAULT_SETTINGS.var_Flipud_frac_INIT
        self.var_Flipud_frac.set(self.var_Flipud_frac_INIT)
        self.var_sometimes_Flipud_frac=tk.StringVar()
        self.var_sometimes_Flipud_frac_INIT=DEFAULT_SETTINGS.var_sometimes_Flipud_frac_INIT
        self.var_sometimes_Flipud_frac.set(self.var_sometimes_Flipud_frac_INIT)


        self.var_Crop=tk.IntVar()
        self.var_Crop_INIT=DEFAULT_SETTINGS.var_Crop_INIT
        self.var_Crop.set(self.var_Crop_INIT)
        self.var_Crop_frac1=tk.StringVar()
        self.var_Crop_frac1_INIT=DEFAULT_SETTINGS.var_Crop_frac1_INIT
        self.var_Crop_frac1.set(self.var_Crop_frac1_INIT)
        self.var_Crop_frac2=tk.StringVar()
        self.var_Crop_frac2_INIT=DEFAULT_SETTINGS.var_Crop_frac2_INIT
        self.var_Crop_frac2.set(self.var_Crop_frac2_INIT)
        self.var_sometimes_Crop_frac=tk.StringVar()
        self.var_sometimes_Crop_frac_INIT=DEFAULT_SETTINGS.var_sometimes_Crop_frac_INIT
        self.var_sometimes_Crop_frac.set(self.var_sometimes_Crop_frac_INIT)


        self.var_Affine=tk.IntVar()
        self.var_Affine_INIT=DEFAULT_SETTINGS.var_Affine_INIT
        self.var_Affine.set(self.var_Affine_INIT)
        self.var_Affine_frac1=tk.StringVar()
        self.var_Affine_frac1_INIT=DEFAULT_SETTINGS.var_Affine_frac1_INIT
        self.var_Affine_frac1.set(self.var_Affine_frac1_INIT)
        self.var_Affine_frac2=tk.StringVar()
        self.var_Affine_frac2_INIT=DEFAULT_SETTINGS.var_Affine_frac2_INIT
        self.var_Affine_frac2.set(self.var_Affine_frac2_INIT)
        self.var_sometimes_Affine_frac=tk.StringVar()
        self.var_sometimes_Affine_frac_INIT=DEFAULT_SETTINGS.var_sometimes_Affine_frac_INIT
        self.var_sometimes_Affine_frac.set(self.var_sometimes_Affine_frac_INIT)

        self.var_GrayScale=tk.IntVar()
        self.var_GrayScale_INIT=DEFAULT_SETTINGS.var_GrayScale_INIT
        self.var_GrayScale.set(self.var_GrayScale_INIT)
        self.var_GrayScale_frac1=tk.StringVar()
        self.var_GrayScale_frac1_INIT=DEFAULT_SETTINGS.var_GrayScale_frac1_INIT
        self.var_GrayScale_frac1.set(self.var_GrayScale_frac1_INIT)
        self.var_sometimes_GrayScale_frac=tk.StringVar()
        self.var_sometimes_GrayScale_frac_INIT=DEFAULT_SETTINGS.var_sometimes_GrayScale_frac_INIT
        self.var_sometimes_GrayScale_frac.set(self.var_sometimes_GrayScale_frac_INIT)

        self.var_ColorTemp=tk.IntVar()
        self.var_ColorTemp_INIT=DEFAULT_SETTINGS.var_ColorTemp_INIT
        self.var_ColorTemp.set(self.var_ColorTemp_INIT)
        self.var_ColorTemp_frac1=tk.StringVar()
        self.var_ColorTemp_frac1_INIT=DEFAULT_SETTINGS.var_ColorTemp_frac1_INIT
        self.var_ColorTemp_frac1.set(self.var_ColorTemp_frac1_INIT)
        self.var_ColorTemp_frac2=tk.StringVar()
        self.var_ColorTemp_frac2_INIT=DEFAULT_SETTINGS.var_ColorTemp_frac2_INIT
        self.var_ColorTemp_frac2.set(self.var_ColorTemp_frac2_INIT)
        self.var_sometimes_ColorTemp_frac=tk.StringVar()
        self.var_sometimes_ColorTemp_frac_INIT=DEFAULT_SETTINGS.var_sometimes_ColorTemp_frac_INIT
        self.var_sometimes_ColorTemp_frac.set(self.var_sometimes_ColorTemp_frac_INIT)

        self.var_GaussianBlur=tk.IntVar()
        self.var_GaussianBlur_INIT=DEFAULT_SETTINGS.var_GaussianBlur_INIT
        self.var_GaussianBlur.set(self.var_GaussianBlur_INIT)
        self.var_GaussianBlur_frac1=tk.StringVar()
        self.var_GaussianBlur_frac1_INIT=DEFAULT_SETTINGS.var_GaussianBlur_frac1_INIT
        self.var_GaussianBlur_frac1.set(self.var_GaussianBlur_frac1_INIT)
        self.var_sometimes_GaussianBlur_frac=tk.StringVar()
        self.var_sometimes_GaussianBlur_frac_INIT=DEFAULT_SETTINGS.var_sometimes_GaussianBlur_frac_INIT
        self.var_sometimes_GaussianBlur_frac.set(self.var_sometimes_GaussianBlur_frac_INIT)

        var_AffineRotate_INIT=1
        var_AffineRotate_frac1_INIT="-45,-30,-15,-10,-5,5,10,15,30,45"
        var_sometimes_AffineRotate_frac_INIT='0.5'
        self.var_AffineRotate=tk.IntVar()
        try:
            self.var_AffineRotate_INIT=DEFAULT_SETTINGS.var_AffineRotate_INIT
        except:
            print('self.var_AffineRotate_INIT NOT FOUND in DEFAULT_SETTINGS')
            self.var_AffineRotate_INIT= var_AffineRotate_INIT
        self.var_AffineRotate.set(self.var_AffineRotate_INIT)
        self.var_AffineRotate_frac1=tk.StringVar()
        try:
            self.var_AffineRotate_frac1_INIT=DEFAULT_SETTINGS.var_AffineRotate_frac1_INIT
        except:
            print('self.var_AffineRotate_frac1_INIT  NOT FOUND in DEFAULT_SETTINGS')
            self.var_AffineRotate_frac1_INIT=var_AffineRotate_frac1_INIT
        self.var_AffineRotate_frac1.set(self.var_AffineRotate_frac1_INIT)
        self.var_sometimes_AffineRotate_frac=tk.StringVar()
        try:
            self.var_sometimes_AffineRotate_frac_INIT=DEFAULT_SETTINGS.var_sometimes_AffineRotate_frac_INIT
        except:
            print('self.var_sometimes_AffineRotate_frac_INIT NOT FOUND in DEFAULT_SETTINGS')
            self.var_sometimes_AffineRotate_frac_INIT=var_sometimes_AffineRotate_frac_INIT
        self.var_sometimes_AffineRotate_frac.set(self.var_sometimes_Affine_frac_INIT)

        var_FakeImage_INIT=1
        var_sometimes_FakeImage_frac_INIT='0.5'
        self.var_FakeImage=tk.IntVar()
        try:
            self.var_FakeImage_INIT=DEFAULT_SETTINGS.var_FakeImage_INIT
        except:
            print('self.var_FakeImage_INIT NOT FOUND in DEFAULT_SETTINGS')
            self.var_FakeImage_INIT= var_FakeImage_INIT
        self.var_FakeImage.set(self.var_FakeImage_INIT)
        self.var_FakeImage_frac1=tk.StringVar()
        self.var_sometimes_FakeImage_frac=tk.StringVar()
        try:
            self.var_sometimes_FakeImage_frac_INIT=DEFAULT_SETTINGS.var_sometimes_FakeImage_frac_INIT
        except:
            print('self.var_sometimes_FakeImage_frac_INIT NOT FOUND in DEFAULT_SETTINGS')
            self.var_sometimes_FakeImage_frac_INIT=var_sometimes_FakeImage_frac_INIT           

        self.var_sometimes_FakeImage_frac.set(self.var_sometimes_FakeImage_frac_INIT)

        self.c_HEADER1_label=tk.Label(self.root,text='Img Aug Selection',bg=self.root_bg,fg=self.root_fg,font=('Arial 14 underline'))
        self.c_HEADER1_label.grid(row=12,column=2,sticky='sw')

        self.c_HEADER2_label=tk.Label(self.root,text='Option 1',bg=self.root_bg,fg=self.root_fg,font=('Arial 10 underline'))
        self.c_HEADER2_label.grid(row=12,column=3,sticky='s')

        self.c_HEADER3_label=tk.Label(self.root,text='Option 2',bg=self.root_bg,fg=self.root_fg,font=('Arial 10 underline'))
        self.c_HEADER3_label.grid(row=12,column=4,sticky='s')

        self.c_HEADER4_label=tk.Label(self.root,text='Sometimes',bg=self.root_bg,fg=self.root_fg,font=('Arial 10 underline'))
        self.c_HEADER4_label.grid(row=12,column=5,sticky='s')

        self.c_sometimes = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Sometimes',variable=self.var_sometimes, onvalue=1, offvalue=0)
        self.c_sometimes.grid(row=13,column=2,sticky='sw')
        self.c_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_frac)
        self.c_sometimes_frac_entry.grid(row=13,column=3,sticky='sw')
        self.c_sometimes_frac_label=tk.Label(self.root,text='Fraction - Sometimes',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_sometimes_frac_label.grid(row=14,column=3,sticky='nw')

        self.c_Fliplr = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Random Horizontal Flip',variable=self.var_Fliplr, onvalue=1, offvalue=0)
        self.c_Fliplr.grid(row=15,column=2,sticky='sw')
        self.c_Fliplr_frac_entry=tk.Entry(self.root,textvariable=self.var_Fliplr_frac)
        self.c_Fliplr_frac_entry.grid(row=15,column=3,sticky='sw')
        self.c_Fliplr_frac_label=tk.Label(self.root,text='Fraction Random Flip - Horizontal',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_Fliplr_frac_label.grid(row=16,column=3,sticky='nw')
        self.c_Fliplr_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_Fliplr_frac)
        self.c_Fliplr_sometimes_frac_entry.grid(row=15,column=5,sticky='se')
        self.c_Fliplr_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_Fliplr_sometimes_frac_label.grid(row=16,column=5,sticky='ne')

        self.c_Flipud = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Random Vertical Flip',variable=self.var_Flipud, onvalue=1, offvalue=0)
        self.c_Flipud.grid(row=17,column=2,sticky='sw')
        self.c_Flipud_frac_entry=tk.Entry(self.root,textvariable=self.var_Flipud_frac)
        self.c_Flipud_frac_entry.grid(row=17,column=3,sticky='sw')
        self.c_Flipud_frac_label=tk.Label(self.root,text='Fraction Random Flip - Vertical',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_Flipud_frac_label.grid(row=18,column=3,sticky='nw')
        self.c_Flipud_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_Flipud_frac)
        self.c_Flipud_sometimes_frac_entry.grid(row=17,column=5,sticky='se')
        self.c_Flipud_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_Flipud_sometimes_frac_label.grid(row=18,column=5,sticky='ne')
    
        self.c_Crop = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Random Crop',variable=self.var_Crop, onvalue=1, offvalue=0)
        self.c_Crop.grid(row=19,column=2,sticky='sw')
        self.c_Crop_frac1_entry=tk.Entry(self.root,textvariable=self.var_Crop_frac1)
        self.c_Crop_frac1_entry.grid(row=19,column=3,sticky='sw')
        self.c_Crop_frac1_label=tk.Label(self.root,text='Percent min',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_Crop_frac1_label.grid(row=20,column=3,sticky='nw')
        self.c_Crop_frac2_entry=tk.Entry(self.root,textvariable=self.var_Crop_frac2)
        self.c_Crop_frac2_entry.grid(row=19,column=4,sticky='sw')
        self.c_Crop_frac2_label=tk.Label(self.root,text='Percent max',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_Crop_frac2_label.grid(row=20,column=4,sticky='nw')
        self.c_Crop_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_Crop_frac)
        self.c_Crop_sometimes_frac_entry.grid(row=19,column=5,sticky='se')
        self.c_Crop_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_Crop_sometimes_frac_label.grid(row=20,column=5,sticky='ne')

        self.c_Affine = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Random Affine',variable=self.var_Affine, onvalue=1, offvalue=0)
        self.c_Affine.grid(row=21,column=2,sticky='sw')
        self.c_Affine_frac1_entry=tk.Entry(self.root,textvariable=self.var_Affine_frac1)
        self.c_Affine_frac1_entry.grid(row=21,column=3,sticky='sw')
        self.c_Affine_frac1_label=tk.Label(self.root,text='Min Zoom',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_Affine_frac1_label.grid(row=22,column=3,sticky='nw')
        self.c_Affine_frac2_entry=tk.Entry(self.root,textvariable=self.var_Affine_frac2)
        self.c_Affine_frac2_entry.grid(row=21,column=4,sticky='sw')
        self.c_Affine_frac2_label=tk.Label(self.root,text='Max Zoom',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_Affine_frac2_label.grid(row=22,column=4,sticky='nw')
        self.c_Affine_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_Affine_frac)
        self.c_Affine_sometimes_frac_entry.grid(row=21,column=5,sticky='se')
        self.c_Affine_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_Affine_sometimes_frac_label.grid(row=22,column=5,sticky='ne')

        self.c_GrayScale = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Gray Scale',variable=self.var_GrayScale, onvalue=1, offvalue=0)
        self.c_GrayScale.grid(row=23,column=2,sticky='sw')
        self.c_GrayScale_frac1_entry=tk.Entry(self.root,textvariable=self.var_GrayScale_frac1)
        self.c_GrayScale_frac1_entry.grid(row=23,column=3,sticky='sw')
        self.c_GrayScale_frac1_label=tk.Label(self.root,text='Max',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_GrayScale_frac1_label.grid(row=24,column=3,sticky='nw')
        self.c_GrayScale_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_GrayScale_frac)
        self.c_GrayScale_sometimes_frac_entry.grid(row=23,column=5,sticky='se')
        self.c_GrayScale_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_GrayScale_sometimes_frac_label.grid(row=24,column=5,sticky='ne')

        self.c_ColorTemp = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Color Temperature',variable=self.var_ColorTemp, onvalue=1, offvalue=0)
        self.c_ColorTemp.grid(row=25,column=2,sticky='sw')
        self.c_ColorTemp_frac1_entry=tk.Entry(self.root,textvariable=self.var_ColorTemp_frac1)
        self.c_ColorTemp_frac1_entry.grid(row=25,column=3,sticky='sw')
        self.c_ColorTemp_frac1_label=tk.Label(self.root,text='Min Temp (Kelvin)',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_ColorTemp_frac1_label.grid(row=26,column=3,sticky='nw')
        self.c_ColorTemp_frac2_entry=tk.Entry(self.root,textvariable=self.var_ColorTemp_frac2)
        self.c_ColorTemp_frac2_entry.grid(row=25,column=4,sticky='sw')
        self.c_ColorTemp_frac2_label=tk.Label(self.root,text='Max Temp (Kelvin)',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_ColorTemp_frac2_label.grid(row=26,column=4,sticky='nw')
        self.c_ColorTemp_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_ColorTemp_frac)
        self.c_ColorTemp_sometimes_frac_entry.grid(row=25,column=5,sticky='se')
        self.c_ColorTemp_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_ColorTemp_sometimes_frac_label.grid(row=26,column=5,sticky='ne')

        self.c_GaussianBlur = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Gaussian Blur',variable=self.var_GaussianBlur, onvalue=1, offvalue=0)
        self.c_GaussianBlur.grid(row=27,column=2,sticky='sw')
        self.c_GaussianBlur_frac1_entry=tk.Entry(self.root,textvariable=self.var_GaussianBlur_frac1)
        self.c_GaussianBlur_frac1_entry.grid(row=27,column=3,sticky='sw')
        self.c_GaussianBlur_frac1_label=tk.Label(self.root,text='Sigma',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_GaussianBlur_frac1_label.grid(row=28,column=3,sticky='nw')
        self.c_GaussianBlur_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_GaussianBlur_frac)
        self.c_GaussianBlur_sometimes_frac_entry.grid(row=27,column=5,sticky='se')
        self.c_GaussianBlur_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_GaussianBlur_sometimes_frac_label.grid(row=28,column=5,sticky='ne')


        self.c_AffineRotate = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Random Affine Rotation',variable=self.var_AffineRotate, onvalue=1, offvalue=0)
        self.c_AffineRotate.grid(row=29,column=2,sticky='sw')
        self.c_AffineRotate_frac1_entry=tk.Entry(self.root,textvariable=self.var_AffineRotate_frac1)
        self.c_AffineRotate_frac1_entry.grid(row=29,column=3,sticky='sw')
        self.c_AffineRotate_frac1_label=tk.Label(self.root,text='Rotation List',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.c_AffineRotate_frac1_label.grid(row=30,column=3,sticky='nw')
        self.c_AffineRotate_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_AffineRotate_frac)
        self.c_AffineRotate_sometimes_frac_entry.grid(row=29,column=5,sticky='se')
        self.c_AffineRotate_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_AffineRotate_sometimes_frac_label.grid(row=30,column=5,sticky='ne')

        self.c_FakeImage = ttk.Checkbutton(self.root, style='Normal.TCheckbutton',text='Fake Background Image',variable=self.var_FakeImage, onvalue=1, offvalue=0)
        self.c_FakeImage.grid(row=31,column=2,sticky='sw')
        self.c_FakeImage_sometimes_frac_entry=tk.Entry(self.root,textvariable=self.var_sometimes_FakeImage_frac)
        self.c_FakeImage_sometimes_frac_entry.grid(row=31,column=5,sticky='se')
        self.c_FakeImage_sometimes_frac_label=tk.Label(self.root,text='Sometimes - Fraction',bg=self.root_fg,fg=self.root_bg,font=('Arial',7))
        self.c_FakeImage_sometimes_frac_label.grid(row=32,column=5,sticky='ne')

        self.load_my_imgs()

        self.MAX_AUGS=DEFAULT_SETTINGS.MAX_AUGS #number of augmentations per class
        self.MAX_AUGS_VAR=tk.StringVar()
        self.MAX_AUGS_VAR.set(self.MAX_AUGS)
        self.MAX_AUGS_entry=tk.Entry(self.root,textvariable=self.MAX_AUGS_VAR)
        self.MAX_AUGS_entry.grid(row=19,column=1,sticky='sw')
        self.MAX_AUGS_label=tk.Label(self.root,text='# Augs per Class',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.MAX_AUGS_label.grid(row=20,column=1,sticky='nw')

        self.labelImg_buttons()

    def save_settings(self,save_root='libs'):
        self.var_sometimes_INIT=self.var_sometimes.get()
        self.var_sometimes_frac_INIT=self.var_sometimes_frac.get()
        self.var_Fliplr_INIT=self.var_Fliplr.get()
        self.var_Fliplr_frac_INIT=self.var_Fliplr_frac.get()
        self.var_sometimes_Fliplr_frac_INIT=self.var_sometimes_Fliplr_frac.get()
        self.var_Flipud_INIT=self.var_Flipud.get()
        self.var_Flipud_frac_INIT=self.var_Flipud.get()
        self.var_sometimes_Flipud_frac_INIT=self.var_sometimes_Flipud_frac.get()
        self.var_Crop_INIT=self.var_Crop.get()
        self.var_Crop_frac1_INIT=self.var_Crop_frac1.get()
        self.var_Crop_frac2_INIT=self.var_Crop_frac2.get()
        self.var_sometimes_Crop_frac_INIT= self.var_sometimes_Crop_frac.get()
        self.var_Affine_INIT=self.var_Affine.get()
        self.var_Affine_frac1_INIT=self.var_Affine_frac1.get()
        self.var_Affine_frac2_INIT=self.var_Affine_frac2.get()
        self.var_sometimes_Affine_frac_INIT=self.var_sometimes_Affine_frac.get()
        self.var_GrayScale_INIT=self.var_GrayScale.get()
        self.var_GrayScale_frac1_INIT= self.var_GrayScale_frac1.get()
        self.var_sometimes_GrayScale_frac_INIT=self.var_sometimes_GrayScale_frac.get()
        self.var_ColorTemp_INIT=self.var_ColorTemp.get()
        self.var_ColorTemp_frac1_INIT=self.var_ColorTemp_frac1.get()
        self.var_ColorTemp_frac2_INIT=self.var_ColorTemp_frac2.get()
        self.var_sometimes_ColorTemp_frac_INIT=self.var_sometimes_ColorTemp_frac.get()
        self.var_GaussianBlur_INIT=self.var_GaussianBlur.get()
        self.var_GaussianBlur_frac1_INIT=self.var_GaussianBlur_frac1.get()
        self.var_sometimes_GaussianBlur_frac_INIT=self.var_sometimes_GaussianBlur_frac.get()
        self.var_AffineRotate_INIT=self.var_AffineRotate.get()
        self.var_AffineRotate_frac1_INIT=self.var_AffineRotate_frac1.get()
        self.var_sometimes_AffineRotate_frac_INIT=self.var_sometimes_AffineRotate_frac.get()
        self.var_FakeImage_INIT=self.var_FakeImage.get()
        self.var_FakeImage_frac1=self.var_FakeImage_frac1.get()

        self.PREFIX=self.PREFIX_VAR.get()
        if os.path.exists('libs/DEFAULT_SETTINGS.py'):
            f=open('libs/DEFAULT_SETTINGS.py','r')
            f_read=f.readlines()
            f.close()
            f_new=[]
            for line in f_read:
                if line.find("path_prefix")==-1 or line.find("format(path_prefix)")!=-1 and line.find('os.path')==-1 and line.find('makedir')==-1:
                    if line.find('=')!=-1:
                        prefix_i=line.split('=')[0]
                        try:
                            prefix_i_comb="self."+prefix_i
                            prefix_i_comb=prefix_i_comb.strip()
                            #print(prefix_i_comb)
                            prefix_i_value=eval(prefix_i_comb)
                        except:
                            pass
                        if line.split('=')[1].find("r'")!=-1:
                            prefix_i_value="r'"+prefix_i_value+"'"
                        elif type(prefix_i_value).__name__.find('int')!=-1:
                            pass
                        elif type(prefix_i_value).__name__.find('str')!=-1:
                            prefix_i_value="'"+prefix_i_value+"'"
                        f_new.append(prefix_i+"="+str(prefix_i_value)+'\n')
            if len(self.PREFIX)>0:
                self.prefix_foldername=self.PREFIX+"_"+self.path_JPEGImages.split('JPEGImages')[0].split('/')[-2].strip().replace(' ','_')
            else:
                self.prefix_foldername=self.path_JPEGImages.split('JPEGImages')[0].split('/')[-2].strip().replace(' ','_')
            prefix_save=_platform+'_'+self.prefix_foldername+'_SAVED_SETTINGS'
            try:
                f_new.append('AUGMENTATION_PATH=r"{}"\n'.format(self.Augmentation_path))
            except:
                pass
            f=open('{}/{}.py'.format(save_root,prefix_save.replace('-','_')),'w')
            wrote=[f.writelines(w) for w in f_new]
            f.close()
    def cleanup(self):
        self.top.destroy()
    def popupWindow_labelImg(self):
        try:
            self.top.destroy()
        except:
            pass
        self.top=tk.Toplevel(self.root)
        self.top.geometry( "{}x{}".format(int(self.root.winfo_screenwidth()*0.95//1.1),int(self.root.winfo_screenheight()*0.95//1.1)) )
        self.top.title('LAUNCH labelImg?')
        self.top.configure(background = 'black')
        self.b=Button(self.top,text='Close',command=self.cleanup,bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.b.grid(row=1,column=0,sticky='se')
        self.submit_LABELIMG=Button(self.top,image=self.icon_labelImg,command=partial(self.open_labelImg,False),bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
        self.submit_LABELIMG.grid(row=0,column=1,sticky='se')
        self.submit_LABELIMG_label=tk.Label(self.top,text="Open Original Dataset",bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.submit_LABELIMG_label.grid(row=1,column=1,sticky='ne')
        if os.path.exists(self.open_aug_jpeg_label_var.get()) and os.path.exists(self.open_aug_anno_label_var.get()):
            self.path_JPEGImages_CUSTOM=self.open_aug_jpeg_label_var.get()
            self.path_Annotations_CUSTOM=self.open_aug_anno_label_var.get()   
            self.submit_LABELIMG_CUSTOM=Button(self.top,image=self.icon_labelImg,command=partial(self.open_labelImg,True),bg=DEFAULT_SETTINGS.root_fg, fg=DEFAULT_SETTINGS.root_bg)
            self.submit_LABELIMG_CUSTOM.grid(row=2,column=1,sticky='se')
            self.submit_LABELIMG_label_CUSTOM=tk.Label(self.top,text="Open Augmented Dataset",bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
            self.submit_LABELIMG_label_CUSTOM.grid(row=3,column=1,sticky='ne')

        
    def labelImg_buttons(self):
        if os.path.exists('libs/labelImg_path.py'):

            self.labelImg_button=Button(self.root,image=self.icon_labelImg,command=self.popupWindow_labelImg,bg=self.root_bg,fg=self.root_fg)
            self.labelImg_button.grid(row=1,column=0,sticky='se')
            self.labelImg_button_note=tk.Label(self.root,text='LabelImg',bg=self.root_bg,fg=self.root_fg,font=("Arial", 9))
            self.labelImg_button_note.grid(row=2,column=0,sticky='ne')  

    def open_labelImg(self,custom):
        from libs import labelImg_path
        from multiprocessing import Process
        self.path_labelImg=labelImg_path.path
        self.path_labelImg_predefined_classes_file=os.path.join(os.path.dirname(self.names_path),'predefined_classes.txt')
        shutil.copy(self.names_path,self.path_labelImg_predefined_classes_file)
        if os.path.exists(self.open_aug_jpeg_label_var.get()) and os.path.exists(self.open_aug_anno_label_var.get()):
            self.path_JPEGImages_CUSTOM=self.open_aug_jpeg_label_var.get()
            self.path_Annotations_CUSTOM=self.open_aug_anno_label_var.get()

        if custom:
            pass
        else:
            self.path_JPEGImages_CUSTOM=self.path_JPEGImages
            self.path_Annotations_CUSTOM=self.path_Annotations
        self.path_labelImg_save_dir=self.path_Annotations_CUSTOM
        self.path_labelImg_image_dir=self.path_JPEGImages_CUSTOM
        #self.PYTHON_PATH="python3"
        if os.path.exists(self.path_labelImg):
            self.cmd_i='{} "{}" "{}" "{}" "{}"'.format(self.PYTHON_PATH ,self.path_labelImg,self.path_labelImg_image_dir,self.path_labelImg_predefined_classes_file,self.path_labelImg_save_dir)
            self.labelImg=Process(target=self.run_cmd,args=(self.cmd_i,)).start()
        else:
            self.popup_text='Please provide a valid labelImg.py path. \n  Current path is: {}'.format(self.path_labelImg)

    def get_update_background_img(self):
        self.image=Image.open(self.root_background_img)
        self.image=self.image.resize((self.root_W,self.root_H),Image.ANTIALIAS)
        self.bg=ImageTk.PhotoImage(self.image)
        self.canvas=tk.Canvas(self.root,width=self.root_W,height=self.root_H)
        self.canvas.grid(row=0,column=0,columnspan=self.canvas_columnspan,rowspan=self.canvas_rowspan,sticky='nw')
        self.canvas.create_image(0,0,image=self.bg,anchor='nw')

    def select_folder(self,folder_i,title_i,var_i=None):
        filetypes=(('All files','*.*'))
        if var_i:
            folder_i=var_i.get()
        if os.path.exists(folder_i):
            self.foldername=fd.askdirectory(title=title_i,
                                        initialdir=folder_i)
        else:
            self.foldername=fd.askdirectory(title=title_i)
        showinfo(title='Selected Folder',
                 message=self.foldername)
        folder_i=self.foldername
        if var_i==self.open_anno_label_var:
            self.anno_selected=True
            var_i.set(folder_i)
            self.open_anno_label.destroy()
            del self.open_anno_label
            cmd_i=open_cmd+" '{}'".format(self.open_anno_label_var.get())
            self.open_anno_label=Button(self.root,textvariable=self.open_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
            #self.open_anno_label=tk.Label(self.root,textvariable=self.open_anno_label_var)
            self.open_anno_label.grid(row=2,column=2,columnspan=50,sticky='sw')
            self.path_Annotations=self.foldername
            print(self.path_Annotations)
        if var_i==self.open_jpeg_label_var:
            self.jpeg_selected=True
            var_i.set(folder_i)
            self.open_jpeg_label.destroy()
            del self.open_jpeg_label
            cmd_i=open_cmd+" '{}'".format(self.open_jpeg_label_var.get())
            self.open_jpeg_label=Button(self.root,textvariable=self.open_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
            #self.open_jpeg_label=tk.Label(self.root,textvariable=self.open_jpeg_label_var)
            self.open_jpeg_label.grid(row=4,column=2,columnspan=50,sticky='sw')
            self.path_JPEGImages=self.foldername
            print(self.path_JPEGImages)

    def run_cmd(self,cmd_i):
        os.system(cmd_i)
    def update_widget(self,widget):
        widget.grid_forget()
    def create_df(self):
        self.df_filename=os.path.join(self.basepath,'df_jpgs_xmls.pkl')
        self.names_path=os.path.join(self.basepath,'names.txt')
        if os.path.exists(self.df_filename) and os.path.exists(self.names_path):
            self.df=pd.read_pickle(self.df_filename)
            if 'Annotations' in self.df.columns:
                if len(list(self.df['Annotations'].unique()))==len(self.Annotations):
                    print('Using Existing df_jpgs_xmls.pkl and names.txt')
        else:
            print('Creating new df_jpgs_xmls.pkl and names.txt')
            self.found_names={}
            i=0
            self.df=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','width','height','label_i','cat_i','JPEGImages','Annotations'])
            for full_anno in tqdm(self.total_annos_list):
                anno=os.path.basename(full_anno)#.split('/')[-1]                
                if anno[0]!='.' and anno.find('.xml')!=-1:
                    img_i_name=anno.split('.xml')[0]
                    path_anno_i=os.path.join(self.path_Annotations,img_i_name+'.xml')
                    path_jpeg_i=os.path.join(self.path_JPEGImages,img_i_name+'.jpg')
                    f=open(path_anno_i,'r')
                    f_read=f.readlines()
                    f.close()
                    parser = etree.XMLParser(encoding=ENCODE_METHOD)
                    xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
                    for size_iter in xmltree.findall('size'):
                        width_i=int(size_iter.find('width').text)
                        height_i=int(size_iter.find('height').text)
                        depth_i=int(size_iter.find('depth').text)
                        imgSize=tuple([height_i,width_i,depth_i])
                    num_objs=[w for w in f_read if w.find('object')!=-1]
                    num_objs=len(num_objs)
                    if num_objs==0:
                        label=''
                        print('No objects found')
                        self.df.at[i,'xmin']=str(0)
                        self.df.at[i,'xmax']=str(width_i)
                        self.df.at[i,'ymin']=str(0)
                        self.df.at[i,'ymax']=str(height_i)
                        self.df.at[i,'width']=imgSize[1]
                        self.df.at[i,'height']=imgSize[0]
                        self.df.at[i,'label_i']=label
                        self.df.at[i,'JPEGImages']=path_jpeg_i
                        self.df.at[i,'Annotations']=path_anno_i
                        i+=1
                    else:
                        for object_iter in xmltree.findall('object'):
                            bndbox = object_iter.find("bndbox")
                            label = object_iter.find('name').text
                            if label not in self.found_names.keys():
                                self.found_names[label]=len(self.found_names.keys())+0
                            xmin = int(float(bndbox.find('xmin').text))
                            ymin = int(float(bndbox.find('ymin').text))
                            xmax = int(float(bndbox.find('xmax').text))
                            ymax = int(float(bndbox.find('ymax').text))
                            self.df.at[i,'xmin']=str(xmin)
                            self.df.at[i,'xmax']=str(xmax)
                            self.df.at[i,'ymin']=str(ymin)
                            self.df.at[i,'ymax']=str(ymax)
                            self.df.at[i,'width']=imgSize[1]
                            self.df.at[i,'height']=imgSize[0]
                            self.df.at[i,'label_i']=label
                            self.df.at[i,'JPEGImages']=path_jpeg_i
                            self.df.at[i,'Annotations']=path_anno_i
                            i+=1
            if len(self.df)>0:
                self.df=self.df.drop_duplicates(keep='last').reset_index().drop('index',axis=1)
                self.df.to_pickle(self.df_filename,protocol=2)
            
            f=open(self.names_path,'w')
            f.writelines([k+'\n' for k, v in sorted(self.found_names.items(), key=lambda item: item[1])])
            f.close()
    def create_post_df(self):
        self.FINAL_TRAIN_FILES=os.listdir(self.TRAIN_PATH_ANNO)
        self.FINAL_TRAIN_FILES=[os.path.join(self.TRAIN_PATH_ANNO,w) for w in self.FINAL_TRAIN_FILES if os.path.exists(os.path.join(self.TRAIN_PATH_ANNO,w))]
        self.found_names_train={}
        i=0
        self.df_train=pd.DataFrame(columns=['xmin','xmax','ymin','ymax','width','height','label_i','cat_i','JPEGImages','Annotations'])
        for full_anno in tqdm(self.FINAL_TRAIN_FILES):
            anno=os.path.basename(full_anno)#.split('/')[-1]                
            if anno[0]!='.' and anno.find('.xml')!=-1:
                img_i_name=anno.split('.xml')[0]
                path_anno_i=os.path.join(self.TRAIN_PATH_ANNO,img_i_name+'.xml')
                path_jpeg_i=os.path.join(self.TRAIN_PATH_JPEG,img_i_name+'.jpg')
                f=open(path_anno_i,'r')
                f_read=f.readlines()
                f.close()
                parser = etree.XMLParser(encoding=ENCODE_METHOD)
                xmltree = ElementTree.parse(path_anno_i, parser=parser).getroot()
                for size_iter in xmltree.findall('size'):
                    width_i=int(size_iter.find('width').text)
                    height_i=int(size_iter.find('height').text)
                    depth_i=int(size_iter.find('depth').text)
                    imgSize=tuple([height_i,width_i,depth_i])
                num_objs=[w for w in f_read if w.find('object')!=-1]
                num_objs=len(num_objs)
                if num_objs==0:
                    label=''
                    print('No objects found')
                    self.df_train.at[i,'xmin']=str(0)
                    self.df_train.at[i,'xmax']=str(width_i)
                    self.df_train.at[i,'ymin']=str(0)
                    self.df_train.at[i,'ymax']=str(height_i)
                    self.df_train.at[i,'width']=imgSize[1]
                    self.df_train.at[i,'height']=imgSize[0]
                    self.df_train.at[i,'label_i']=label
                    self.df_train.at[i,'JPEGImages']=path_jpeg_i
                    self.df_train.at[i,'Annotations']=path_anno_i
                    i+=1
                else:
                    for object_iter in xmltree.findall('object'):
                        bndbox = object_iter.find("bndbox")
                        label = object_iter.find('name').text
                        if label not in self.found_names_train.keys():
                            self.found_names_train[label]=len(self.found_names_train.keys())+0
                        xmin = int(float(bndbox.find('xmin').text))
                        ymin = int(float(bndbox.find('ymin').text))
                        xmax = int(float(bndbox.find('xmax').text))
                        ymax = int(float(bndbox.find('ymax').text))
                        self.df_train.at[i,'xmin']=str(xmin)
                        self.df_train.at[i,'xmax']=str(xmax)
                        self.df_train.at[i,'ymin']=str(ymin)
                        self.df_train.at[i,'ymax']=str(ymax)
                        self.df_train.at[i,'width']=imgSize[1]
                        self.df_train.at[i,'height']=imgSize[0]
                        self.df_train.at[i,'label_i']=label
                        self.df_train.at[i,'JPEGImages']=path_jpeg_i
                        self.df_train.at[i,'Annotations']=path_anno_i
                        i+=1

    def load_my_imgs(self):
        print("LOADING IMGS")
        self.basepath=os.path.dirname(self.path_Annotations)#.split('Annotations')[0]
        self.BACKUP_basepath=os.path.join(self.basepath,'BACKUP')
        self.BACKUP_Annotations=os.path.join(self.BACKUP_basepath,'Annotations')
        self.BACKUP_JPEGImages=os.path.join(self.BACKUP_basepath,'JPEGImages')
        if os.path.exists(self.BACKUP_basepath)==False:
            os.makedirs(self.BACKUP_basepath)
        if os.path.exists(self.BACKUP_Annotations)==False:
            os.makedirs(self.BACKUP_Annotations)
        if os.path.exists(self.BACKUP_JPEGImages)==False:
            os.makedirs(self.BACKUP_JPEGImages)
        if os.path.exists(self.BACKUP_Annotations):
            
            backup_annos=os.listdir(self.BACKUP_Annotations)
            backup_annos=[os.path.join(self.BACKUP_Annotations,w) for w in backup_annos if w.find('.xml')!=-1]
            if len(backup_annos)>0:
                print('MOVING BACKUP_Annotations to Annotations')
                for anno in tqdm(backup_annos):
                    if anno.find('.xml')!=-1:
                        shutil.move(anno,self.path_Annotations)
        if os.path.exists(self.BACKUP_JPEGImages):
            
            backup_jpegs=os.listdir(self.BACKUP_JPEGImages)
            backup_jpegs=[os.path.join(self.BACKUP_JPEGImages,w) for w in backup_jpegs if w.find('.jpg')!=-1]
            if len(backup_jpegs)>0:
                print('MOVING BACKUP_JPEGImages to JPEGImages')
                for jpg in tqdm(backup_jpegs):
                    if jpg.find('.jpg')!=-1:
                        shutil.move(jpg,self.path_JPEGImages)
        try:
            self.TRAIN_SPLIT=int(float(self.TRAIN_SPLIT_VAR.get()))
        except:
            print("EXCEPTION, using 70%")
            self.TRAIN_SPLIT=70
        if self.TRAIN_SPLIT>99:
            self.TRAIN_SPLIT=99
            self.TRAIN_SPLIT_VAR.set(self.TRAIN_SPLIT)
        elif self.TRAIN_SPLIT<0:
            self.TRAIN_SPLIT=1
            self.TRAIN_SPLIT_VAR.set(self.TRAIN_SPLIT)   
        self.Annotations=os.listdir(self.path_Annotations)
        self.Annotations=[os.path.join(self.path_Annotations,w) for w in self.Annotations if w.find('.xml')!=-1 and os.path.exists(os.path.join(self.path_Annotations,w).replace('Annotations','JPEGImages').replace('.xml','.jpg'))]
        self.JPEGImages=[w.replace('Annotations','JPEGImages').replace('.xml','.jpg') for w in self.Annotations]


        self.total_annos_list=self.Annotations
        self.create_df()
        f=open(self.names_path,'r')
        f_read=f.readlines()
        f.close()
        self.unique_names=[w.replace('\n','').strip() for w in f_read]

        self.TRAIN_LIST=[]
        self.TEST_LIST=[]
        self.label_counter_before={}
        self.total_list_i=None
        for unique_label in tqdm(self.unique_names):
            self.unique_label_count_train=0
            self.unique_label_count_test=0
            current_jpegs=os.listdir(self.path_JPEGImages)
            current_jpegs=[os.path.join(self.path_JPEGImages,w) for w in current_jpegs if w.find('.jpg')!=-1]
            self.df=self.df[self.df['JPEGImages'].isin(current_jpegs)].reset_index().drop('index',axis=1)
            self.df_i=self.df[self.df['label_i']==unique_label].copy()
            print('NUMBER OF OBJECTS FOR "{}" == "{}"'.format(unique_label,len(self.df_i)))
            self.df_i=self.df_i.sample(frac=1,random_state=42) #shuffle all rows
            self.df_i=self.df_i.drop_duplicates().sort_values(by='JPEGImages').reset_index().drop('index',axis=1)
            #Limit the number of images on a per class basis
            self.MAX_KEEP=int(self.MAX_KEEP_VAR.get())
            if len(self.df_i)>self.MAX_KEEP:
                tmp_JPGs=list(self.df_i['JPEGImages'].copy())
                for jpg_i in tmp_JPGs:
                    if jpg_i in self.TRAIN_LIST or jpg_i in self.TEST_LIST:
                        self.df_i=self.df_i[self.df_i['JPEGImages']!=jpg_i].reset_index().drop('index',axis=1)
                        
                #self.df_i['JPEGImages']=[w for w in self.df_i['JPEGImages'] if w not in self.TRAIN_LIST and w not in self.TEST_LIST]
                if len(self.df_i)>self.MAX_KEEP:
                    #self.df_i=self.df_i.sample(n=self.MAX_KEEP,random_state=42) #random fraction set to MAX_KEEP
                    #move images to directory called exceed number that are past the max_keep
                    
                    df_backup=self.df_i.loc[self.MAX_KEEP:].reset_index().drop('index',axis=1)
                    backup_jpegs=list(df_backup['JPEGImages'])
                    backup_annos=[w.replace('JPEGImages','Annotations').replace('.jpg','.xml') for w in backup_jpegs if w.find('.jpg')!=-1 and os.path.exists(w)]
                    if len(backup_jpegs)>0:
                        print('Moving extra jpegs to BACKUP_JPEGImages')
                        for jpg in tqdm(backup_jpegs):
                            if os.path.exists(os.path.join(self.BACKUP_JPEGImages,os.path.basename(jpg))):
                                os.remove(os.path.join(self.BACKUP_JPEGImages,os.path.basename(jpg)))
                            try:
                                shutil.move(jpg,self.BACKUP_JPEGImages)
                            except:
                                print('ERROR moving {}'.format(jpg))
                    if len(backup_annos)>0:
                        print('Moving extra annos to BACKUP_Annotations')
                        for anno in tqdm(backup_annos):
                            if os.path.exists(os.path.join(self.BACKUP_Annotations,os.path.basename(anno))):
                                os.remove(os.path.join(self.BACKUP_Annotations,os.path.basename(anno)))
                            try:
                                shutil.move(anno,self.BACKUP_Annotations)
                            except:
                                print('ERROR moving {}'.format(anno))
                    self.df_i=self.df_i.loc[0:self.MAX_KEEP]
                else:
                    tmp_len=len(self.df_i)
                    print('RAN OUT OF IMAGES TO SAMPLE FROM')
                    self.df_i=self.df_i.sample(n=tmp_len,random_state=42) #random fraction set to MAX_KEEP
            else: 
                self.df_i=self.df_i.sample(frac=1,random_state=42) #shuffle all rows
            print('NUMBER OF OBJECTS AFTER "{}" == "{}"'.format(unique_label,len(self.df_i)))
            self.df_i=self.df_i.sort_values(by='JPEGImages')
            self.total_list_i=list(self.df_i['JPEGImages'])
            self.train_list_i=self.total_list_i[:int(self.TRAIN_SPLIT*len(self.df_i)/100.)]
            self.test_list_i=self.total_list_i[int(self.TRAIN_SPLIT*len(self.df_i)/100.):]
            self.TRAIN_LIST+=self.train_list_i
            self.TEST_LIST+=self.test_list_i
            self.unique_label_count_train+=len(self.train_list_i)
            self.unique_label_count_test+=len(self.test_list_i)
            self.label_counter_before[unique_label]=[self.unique_label_count_train,self.unique_label_count_test]
        self.TEST_LIST=list(pd.DataFrame(self.TEST_LIST)[0].drop_duplicates())
        self.TRAIN_LIST=list(pd.DataFrame(self.TRAIN_LIST)[0].drop_duplicates())    
        print(len(self.TEST_LIST)+len(self.TRAIN_LIST))
        #self.VAL_LIST=set(self.VAL_LIST)-set(self.TRAIN_LIST)
        if self.TRAIN_SPLIT<50:
            self.TRAIN_LIST=set(self.TRAIN_LIST)-set(self.TEST_LIST)
        else:
            self.TRAIN_LIST=set(self.TRAIN_LIST)-set(self.TEST_LIST)
        print('\nIMAGE COUNTS')
        print('len(self.TEST_LIST) =',len(self.TEST_LIST))
        print('len(self.TRAIN_LIST) =',len(self.TRAIN_LIST))
        print('len(self.TEST_LIST)+len(self.TRAIN_LIST) = ',len(self.TEST_LIST)+len(self.TRAIN_LIST))

        self.train_list_jpegs=self.TRAIN_LIST
        self.test_list_jpegs=self.TEST_LIST

        self.train_list_annos=[w.replace('JPEGImages','Annotations').replace('.jpg','.xml') for w in self.train_list_jpegs if os.path.exists(w.replace('JPEGImages','Annotations').replace('.jpg','.xml'))]
        self.test_list_annos=[w.replace('JPEGImages','Annotations').replace('.jpg','.xml') for w in self.test_list_jpegs if os.path.exists(w.replace('JPEGImages','Annotations').replace('.jpg','.xml'))]

        print('\nOBJECT COUNTS')
        self.max_count=0
        self.label_counter={}
        try:
            if len(self.TRAIN_label_dic)>0:
                for item in self.TRAIN_label_dic.values():
                    try:
                        item.destroy()
                    except:
                        pass
        except:
            pass
        try:
            if len(self.TRAIN_count_dic)>0:
                for item in self.TRAIN_count_dic.values():
                    try:
                        item.destroy()
                    except:
                        pass
        except:
            pass
        try:
            if len(self.TEST_count_dic)>0:
                for item in self.TEST_count_dic.values():
                    try:
                        item.destroy()
                    except:
                        pass
        except:
            pass
        try:
            if len(self.TRAIN_POST_count_dic)>0:
                for item in self.TRAIN_POST_count_dic.values():
                    try:
                        item.destroy()
                    except:
                        pass
        except:
            pass
        try:
            self.LABEL_title.destroy()
        except:
            pass
        try:
            self.traincount_title.destroy()
        except:
            pass
        try:
            self.testcount_title.destroy()
        except:
            pass
        self.TRAIN_label_dic={}
        self.TRAIN_count_dic={}
        self.TEST_count_dic={}
        self.TRAIN_POST_count_dic={}
        i=1
        j=2


        self.LABEL_title=tk.Label(self.root,text='Label',bg=self.root_fg,fg=self.root_bg,font=('Arial 14 underline'))
        self.LABEL_title.grid(row=i,column=j+6,sticky='nw')
        self.traincount_title=tk.Label(self.root,text='Train Count OG #',bg=self.root_fg,fg=self.root_bg,font=('Arial 14 underline'))
        self.traincount_title.grid(row=i,column=j+7,sticky='nw')
        self.testcount_title=tk.Label(self.root,text='Test Count OG #',bg=self.root_fg,fg=self.root_bg,font=('Arial 14 underline'))
        self.testcount_title.grid(row=i,column=j+8,sticky='nw')

        for label,(count_train,count_test) in self.label_counter_before.items():
            if count_train>self.max_count:
                self.max_count=count_train
            self.label_counter[label]=count_train
            print("LABEL={}; TRAIN={}; TEST={}".format(label,count_train,count_test))
            self.TRAIN_label_dic[label]=tk.Label(self.root,text=label,bg=self.root_bg,fg=self.root_fg,font=('Arial 10 underline'))
            self.TRAIN_label_dic[label].grid(row=i+1,column=j+6,sticky='nw')
            self.TRAIN_count_dic[label]=tk.Label(self.root,text=count_train,bg=self.root_bg,fg=self.root_fg,font=('Arial 10 bold'))
            self.TRAIN_count_dic[label].grid(row=i+1,column=j+7,sticky='nw')
            self.TEST_count_dic[label]=tk.Label(self.root,text=count_test,bg=self.root_bg,fg=self.root_fg,font=('Arial 10 bold'))
            self.TEST_count_dic[label].grid(row=i+1,column=j+8,sticky='nw')
            i+=1

        try:
            self.MAX_AUGS=int(float(self.MAX_AUGS_VAR.get()))
            if self.MAX_AUGS<self.max_count:
                self.MAX_AUGS=self.max_count
                self.MAX_AUGS_VAR.set(self.max_count)
            else:
                self.max_count=self.MAX_AUGS
            print('using self.MAX_AUGS = {} per class \n'.format(self.max_count))
        except:
            print('using self.max_counts = {} per class \n'.format(self.max_count))

        self.TRAIN_label_dic={}
        
        self.TRAIN_SPLIT_label=tk.Label(self.root,text='TRAIN SPLIT',bg=self.root_bg,fg=self.root_fg,font=('Arial',7))
        self.TRAIN_SPLIT_label.grid(row=16,column=1,sticky='nw')

    def put_fake_background(self,img,bbs):
        path_background='FAKE_BACKGROUNDS'
        if os.path.exists(path_background):
            possible_fake_backgrounds=os.listdir(path_background)
            if len(possible_fake_backgrounds)>0:
                possible_fake_backgrounds=[os.path.join(path_background,w) for w in possible_fake_backgrounds if os.path.isfile(os.path.join(path_background,w)) and w.find('.jpg')!=-1]
                if len(possible_fake_backgrounds)>0:
                    path_background_i=np.random.choice(possible_fake_backgrounds)
                    img_og=img
                    for i,bb in enumerate(bbs):
                        #print('path_background_i',path_background_i)
                        #print('bbs',bb)
                        xmin=int(bb.x1)
                        xmax=int(bb.x2)
                        ymin=int(bb.y1)
                        ymax=int(bb.y2)
                        #print('xmin','ymin','xmax','ymax')
                        #print(xmin,ymin,xmax,ymax)
                        img_i=img_og
                        img_i_H=img_i.shape[1]
                        img_i_W=img_i.shape[0]
                        if i==0:
                            img_fake=cv2.imread(path_background_i)
                            img_fake=cv2.resize(img_fake,(img_i_H,img_i_W))
                        if len(img_i.shape)==3:
                            img_fake[ymin:ymax,xmin:xmax,:]=img_i[ymin:ymax,xmin:xmax,:]
                        else:
                            img_fake[ymin:ymax,xmin:xmax]=img_i[ymin:ymax,xmin:xmax]
                        img=img_fake
        return img,bbs
        
    def augment_my_imgs(self):
        self.load_my_imgs()


        self.Augmentation_path=os.path.join(self.basepath,'Augmentations')
        if os.path.exists(self.Augmentation_path)==False:
            os.makedirs(self.Augmentation_path)
        else:
            remove_directory(self.Augmentation_path)
            os.makedirs(self.Augmentation_path)
        self.TRAIN_PATH=os.path.join(self.Augmentation_path,'train')
        self.TEST_PATH=os.path.join(self.Augmentation_path,'test')
        print('\nOriginal TRAIN Count = {}\n'.format(len(self.train_list_annos)))
        print('Original TEST Count = {}\n'.format(len(self.test_list_annos)))
        if os.path.exists(self.TRAIN_PATH)==False:
            os.makedirs(self.TRAIN_PATH)
        else:
            remove_directory(self.TRAIN_PATH)
            os.makedirs(self.TRAIN_PATH)
        if os.path.exists(self.TEST_PATH)==False:
            os.makedirs(self.TEST_PATH)
        else:
            remove_directory(self.TEST_PATH)
            os.makedirs(self.TEST_PATH)
        self.TEST_PATH_ANNO=os.path.join(self.TEST_PATH,'Annotations')
        self.TEST_PATH_JPEG=os.path.join(self.TEST_PATH,'JPEGImages')
        if os.path.exists(self.TEST_PATH_ANNO)==False:
            os.makedirs(self.TEST_PATH_ANNO)
        else:
            remove_directory(self.TEST_PATH_ANNO)
            os.makedirs(self.TEST_PATH_ANNO)
        if os.path.exists(self.TEST_PATH_JPEG)==False:
            os.makedirs(self.TEST_PATH_JPEG)
        else:
            remove_directory(self.TEST_PATH_JPEG)
            os.makedirs(self.TEST_PATH_JPEG)

        for test_anno,test_jpg in tqdm(zip(self.test_list_annos,self.test_list_jpegs)):
            try:
                shutil.copy(test_anno,self.TEST_PATH_ANNO)
            except:
                print('ERROR in moving: \n test_anno={}'.format(test_anno))
            try:
                shutil.copy(test_jpg,self.TEST_PATH_JPEG)
            except:
                print('ERROR in moving: \n test_jpg={}'.format(test_jpg))

        self.TRAIN_PATH_ANNO=os.path.join(self.TRAIN_PATH,'Annotations')
        self.TRAIN_PATH_JPEG=os.path.join(self.TRAIN_PATH,'JPEGImages')
        if os.path.exists(self.TRAIN_PATH_ANNO)==False:
            os.makedirs(self.TRAIN_PATH_ANNO)
        else:
            remove_directory(self.TRAIN_PATH_ANNO)
            os.makedirs(self.TRAIN_PATH_ANNO)
        if os.path.exists(self.TRAIN_PATH_JPEG)==False:
            os.makedirs(self.TRAIN_PATH_JPEG)
        else:
            remove_directory(self.TRAIN_PATH_JPEG)
            os.makedirs(self.TRAIN_PATH_JPEG)
        for train_anno,train_jpg in tqdm(zip(self.train_list_annos,self.train_list_jpegs)):
            try:
                shutil.copy(train_anno,self.TRAIN_PATH_ANNO)
            except:
                print('ERROR in moving: \n train_anno={}'.format(test_anno))      
            try:
                shutil.copy(train_jpg,self.TRAIN_PATH_JPEG)
            except:
                print('ERROR in moving: \n train_jpg={}'.format(test_jpg))             

        self.Annotations_aug_path=self.TRAIN_PATH_ANNO
        self.JPEGImages_aug_path=self.TRAIN_PATH_JPEG

        self.open_aug_anno_label_var=tk.StringVar()
        self.open_aug_anno_label_var.set(self.Annotations_aug_path)
        self.open_aug_anno_note=tk.Label(self.root,text="2.a \n Aug \n Annotations dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_aug_anno_note.grid(row=9,column=1,sticky='ne')
        cmd_i=open_cmd+" '{}'".format(self.open_aug_anno_label_var.get())
        self.open_aug_anno_label=Button(self.root,textvariable=self.open_aug_anno_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_aug_anno_label.grid(row=8,column=2,columnspan=50,sticky='sw')
        self.open_aug_anno_button=Button(self.root,image=self.icon_single_file,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_aug_anno_button.grid(row=8,column=1,sticky='se')

        self.open_aug_jpeg_label_var=tk.StringVar()
        self.open_aug_jpeg_label_var.set(self.JPEGImages_aug_path)
        self.open_aug_jpeg_note=tk.Label(self.root,text="2.b \n Aug \n JPEGImages dir",bg=self.root_bg,fg=self.root_fg,font=("Arial", 8))
        self.open_aug_jpeg_note.grid(row=11,column=1,sticky='ne')

        cmd_i=open_cmd+" '{}'".format(self.open_aug_jpeg_label_var.get())
        self.open_aug_jpeg_label=Button(self.root,textvariable=self.open_aug_jpeg_label_var, command=partial(self.run_cmd,cmd_i),bg=self.root_fg,fg=self.root_bg,font=("Arial", 8))
        self.open_aug_jpeg_label.grid(row=10,column=2,columnspan=50,sticky='sw')
        self.open_aug_jpeg_button=Button(self.root,image=self.icon_single_file,command=partial(self.run_cmd,cmd_i),bg=self.root_bg,fg=self.root_fg)
        self.open_aug_jpeg_button.grid(row=10,column=1,sticky='se')
        self.Annotations=[os.path.basename(w) for w in self.train_list_annos]
        self.df['Annotation_basepath']=[os.path.basename(w) for w in self.df['Annotations']]
        self.df['JPEGImages_basepath']=[os.path.basename(w) for w in self.df['JPEGImages']]

        self.df_j=self.df[self.df['Annotation_basepath'].isin(self.Annotations)].copy()
        self.df_j['Annotations']=[os.path.join(self.Annotations_aug_path,w) for w in self.df_j['Annotation_basepath'] if w.find('.xml')!=-1]
        self.df_j['JPEGImages']=[os.path.join(self.JPEGImages_aug_path,w) for w in self.df_j['JPEGImages_basepath'] if w.find('.jpg')!=-1]
        print('LENGTH OF self.df_j=={}'.format(len(self.df_j)))
                

                
        self.Augment_Count={}
        #self.label_counter={}
        for unique_label in tqdm(self.unique_names):
            print('\n\nUNIQUE LABEL = {}\n'.format(unique_label))
            if unique_label not in self.label_counter:
                self.label_counter[unique_label]=0
            unique_label_count_train=0
            self.df_i=self.df_j[self.df_j['label_i']==unique_label].copy()
            self.df_i=self.df_i.drop_duplicates().reset_index().drop('index',axis=1)
            self.df_i=self.df_i.sample(frac=1) #shuffle all rows
            while unique_label_count_train<self.max_count:
                start_count=unique_label_count_train
                #print('Current Count = {}, Desired Count = {}\n'.format(unique_label_count_train,max_count))
                for j,(JPEG_i_path,ANNO_i_path) in tqdm(enumerate(zip(self.df_i['JPEGImages'],self.df_i['Annotations']))):
                    ia.seed(j)
                    ANNO=PascalVocReader(ANNO_i_path)
                    ANNO_i=ANNO.get_shapes()
                    self.img_i=cv2.imread(JPEG_i_path)
                    self.img_i_og=self.img_i
                    if ANNO_i:
                        df_i=pd.DataFrame(columns=['nameA','nameB','boxA','boxB','iou'])
                        jj=0
                        self.bbs=[]
                        self.names=[]
                        for ANNO_j in ANNO_i:
                            #print(gt_i)
                            nameA=ANNO_j[0].strip()
                            #print(nameA)
                            if self.PREFIX_VAR.get()!='':
                                nameA=self.PREFIX_VAR.get()+"_"+nameA
                            else:
                                #print('blank')
                                pass
                            boxA=BoundingBox(x1=ANNO_j[1][0],x2=ANNO_j[1][2],y1=ANNO_j[1][1],y2=ANNO_j[1][3],label=nameA)
                            if ANNO_j[0]!=unique_label and ANNO_j[0] in self.label_counter.keys():
                                if self.label_counter[ANNO_j[0]] > self.max_count:
                                    if len(self.img_i.shape)==3:
                                        self.img_i[ANNO_j[1][1]:ANNO_j[1][3],ANNO_j[1][0]:ANNO_j[1][2],:]=0.
                                    else:
                                        self.img_i[ANNO_j[1][1]:ANNO_j[1][3],ANNO_j[1][0]:ANNO_j[1][2]]=0.
                                    # unique_count_i=len(self.df_j[(self.df_j['label_i']==ANNO_j[0]) & (self.df_j['JPEGImages']==JPEG_i_path)])
                                    # curr_value=self.label_counter[ANNO_j[0]]
                                    # curr_value-=unique_count_i
                                    # self.label_counter[ANNO_j[0]]=curr_value
                                else:
                                    self.names.append(nameA)
                                    self.bbs.append(boxA)
                            elif ANNO_j[0]==unique_label:
                                self.names.append(nameA)
                                self.bbs.append(boxA)
                        self.bbs=BoundingBoxesOnImage(self.bbs,shape=self.img_i.shape)
                        self.augmentations_i()
                        self.image_aug=self.img_i
                        self.bbs_aug=self.bbs
                        #ia.imshow(self.bbs_aug.draw_on_image(self.image_aug,size=2))
                        #print(self.bbs_aug)
                        if np.sum(np.sum(self.img_i-self.img_i_og))!=0:
                            time_i=str(time.time()).split('.')[0]
                            self.JPEG_i_path_aug=os.path.join(self.JPEGImages_aug_path,'aug_'+time_i+'_'+os.path.basename(JPEG_i_path))
                            self.Annotation_i_path_aug=os.path.join(self.Annotations_aug_path,'aug_'+time_i+'_'+os.path.basename(ANNO_i_path))

                            count_i=writePascalVOV(self.image_aug,self.bbs_aug,JPEG_i_path,ANNO_i_path,self.JPEG_i_path_aug,self.Annotation_i_path_aug)

                            if count_i>0:
                                
                                for label_i in self.df_j[self.df_j['JPEGImages']==JPEG_i_path]['label_i'].unique():
                                    unique_count_i=len(self.df_j[(self.df_j['label_i']==label_i) & (self.df_j['JPEGImages']==JPEG_i_path)])
                                    if label_i in self.label_counter.keys():
                                        curr_value=self.label_counter[label_i]
                                        curr_value+=unique_count_i
                                        self.label_counter[label_i]=curr_value
                                    else:
                                        self.label_counter[label_i]=unique_count_i

                                    unique_label_count_train=self.label_counter[unique_label]
                                self.Augment_Count[JPEG_i_path]=JPEG_i_path
                            if unique_label_count_train>=self.max_count:
                                break

                if start_count==unique_label_count_train:
                    print('Issue getting augmentations for: {}'.format(unique_label))
                    break
        self.Augment_Count=len(list(self.Augment_Count.values()))
        print('\n Total Augmentations = {}\n'.format(self.Augment_Count))
        print('\nFinal TRAIN Count = {}\n'.format(len(self.train_list_annos)+self.Augment_Count))
        print('Final TEST Count = {}\n'.format(len(self.test_list_annos)))
        
        self.create_post_df()
        print('Training Label Summary:\n')
        self.label_counter_after={}
        for unique_label in self.df_train['label_i'].unique():
            print('\tLabel = {} has {} instances\n'.format(unique_label,len(self.df_train[self.df_train['label_i']==unique_label])))
            if unique_label in self.label_counter_before.keys():
                self.label_counter_after[unique_label]=[self.label_counter_before[unique_label][0],self.label_counter_before[unique_label][1],len(self.df_train[self.df_train['label_i']==unique_label])]
            else:
                self.label_counter_after[unique_label]=['','',len(self.df_train[self.df_train['label_i']==unique_label])]
            if unique_label in self.label_counter_before.keys():
                print('\t\t original had {} instances\n\n'.format(self.label_counter_before[unique_label]))
        try:
            if len(self.TRAIN_label_dic)>0:
                for item in self.TRAIN_label_dic.values():
                    try:
                        item.destroy()
                    except:
                        pass
        except:
            pass
        try:
            if len(self.TRAIN_count_dic)>0:
                for item in self.TRAIN_count_dic.values():
                    try:
                        item.destroy()
                    except:
                        pass
        except:
            pass
        try:
            if len(self.TEST_count_dic)>0:
                for item in self.TEST_count_dic.values():
                    try:
                        item.destroy()
                    except:
                        pass
        except:
            pass
        try:
            if len(self.TRAIN_POST_count_dic)>0:
                for item in self.TRAIN_POST_count_dic.values():
                    try:
                        item.destroy()
                    except:
                        pass
        except:
            pass
        try:
            self.LABEL_title.destroy()
        except:
            pass
        try:
            self.traincount_title.destroy()
        except:
            pass
        try:
            self.testcount_title.destroy()
        except:
            pass
        try:
            self.traincountafter_title.destroy()
        except:
            pass
        self.TRAIN_label_dic={}
        self.TRAIN_count_dic={}
        self.TEST_count_dic={}
        self.TRAIN_POST_count_dic={}
        i=1
        j=2

        self.LABEL_title=tk.Label(self.root,text='Label',bg=self.root_fg,fg=self.root_bg,font=('Arial 14 underline'))
        self.LABEL_title.grid(row=i,column=j+6,sticky='nw')
        self.traincount_title=tk.Label(self.root,text='Train Count OG #',bg=self.root_fg,fg=self.root_bg,font=('Arial 14 underline'))
        self.traincount_title.grid(row=i,column=j+7,sticky='nw')
        self.testcount_title=tk.Label(self.root,text='Test Count OG #',bg=self.root_fg,fg=self.root_bg,font=('Arial 14 underline'))
        self.testcount_title.grid(row=i,column=j+8,sticky='nw')
        self.traincountafter_title=tk.Label(self.root,text='Train Count w/ AUGS #',bg=self.root_fg,fg=self.root_bg,font=('Arial 14 underline'))
        self.traincountafter_title.grid(row=i,column=j+9,sticky='nw')

        for label,(count_train,count_test,count_train_after) in self.label_counter_after.items():
            print("LABEL={}; TRAIN={}; TEST={}; POST-TRAIN={}".format(label,count_train,count_test,count_train_after))
            try:
                self.TRAIN_label_dic[label].destroy()
            except:
                pass
            self.TRAIN_label_dic[label]=tk.Label(self.root,text=label,bg=self.root_bg,fg=self.root_fg,font=('Arial 10 underline'))
            self.TRAIN_label_dic[label].grid(row=i+1,column=j+6,sticky='nw')
            try:
                self.TRAIN_count_dic[label].destroy()
            except:
                pass
            self.TRAIN_count_dic[label]=tk.Label(self.root,text=count_train,bg=self.root_bg,fg=self.root_fg,font=('Arial 10 bold'))
            self.TRAIN_count_dic[label].grid(row=i+1,column=j+7,sticky='nw')
            try:
                self.TEST_count_dic[label].destroy()
            except:
                pass
            self.TEST_count_dic[label]=tk.Label(self.root,text=count_test,bg=self.root_bg,fg=self.root_fg,font=('Arial 10 bold'))
            self.TEST_count_dic[label].grid(row=i+1,column=j+8,sticky='nw')
            try:
                self.TRAIN_POST_count_dic[label].destroy()
            except:
                pass
            self.TRAIN_POST_count_dic[label]=tk.Label(self.root,text=count_train_after,bg=self.root_bg,fg=self.root_fg,font=('Arial 10 bold'))
            self.TRAIN_POST_count_dic[label].grid(row=i+1,column=j+9,sticky='nw')
            i+=1
        self.save_settings()


    def augmentations_i(self):
        if self.var_sometimes.get()==1:
                    try:
                        sometimes_frac=float(self.var_sometimes_frac.get())
                    except:
                        print('Exception using 0.5 for sometimes fraction')
                        sometimes_frac=float(0.5)  
        if self.var_Fliplr.get()==1:
            try:
                flip_frac=float(self.var_Fliplr_frac.get())
            except:
                print('Exception using 0.5 for Fliplr_frac')
                flip_frac=float(0.5)
            if self.var_sometimes.get()!=1:
                self.img_i,self.bbs=iaa.Fliplr(flip_frac)(image=self.img_i,bounding_boxes=self.bbs)
            else:
                if self.var_sometimes_Fliplr_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_Fliplr=float(self.var_sometimes_Fliplr_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_Fliplr=sometimes_frac
                else:
                    sometimes_frac_Fliplr=sometimes_frac
                self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_Fliplr,iaa.Fliplr(flip_frac))(image=self.img_i,bounding_boxes=self.bbs)
        if self.var_Flipud.get()==1:
            try:
                flip_frac=float(self.var_Flipud_frac.get())
            except:
                print('Exception using 0.5 for Flipud_frac')
                flip_frac=float(0.5)
            if self.var_sometimes.get()!=1:
                self.img_i,self.bbs=iaa.Flipud(flip_frac)(image=self.img_i,bounding_boxes=self.bbs)
            else:
                if self.var_sometimes_Flipud_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_Flipud=float(self.var_sometimes_Flipud_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_Flipud=sometimes_frac
                else:
                    sometimes_frac_Flipud=sometimes_frac
                self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_Flipud,iaa.Flipud(flip_frac))(image=self.img_i,bounding_boxes=self.bbs)

        if self.var_Crop.get()==1:
            try:
                crop_frac1=float(self.var_Crop_frac1.get())
            except:
                print('Exception using 0 for Crop_frac1')
                crop_frac1=float(0)
            try:
                crop_frac2=float(self.var_Crop_frac2.get())
            except:
                print('Exception using 0.5 for Crop_frac2')
                crop_frac2=float(0.5)
            if self.var_sometimes.get()!=1:
                self.img_i,self.bbs=iaa.Crop(percent=(crop_frac1,crop_frac2))(image=self.img_i,bounding_boxes=self.bbs)
            else:
                if self.var_sometimes_Crop_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_Crop=float(self.var_sometimes_Crop_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_Crop=sometimes_frac
                else:
                    sometimes_frac_Crop=sometimes_frac
                self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_Crop,iaa.Crop(percent=(crop_frac1,crop_frac2)))(image=self.img_i,bounding_boxes=self.bbs)

        if self.var_Affine.get()==1:
            try:
                scale_frac1=float(self.var_Affine_frac1.get())
            except:
                print('Exception using 0.5 for min scale for Affine_frac')
                scale_frac1=float(0.5)
            try:
                scale_frac2=float(self.var_Affine_frac2.get())
            except:
                print('Exception using 1.5 for max scale for Affine_frac')
                scale_frac2=float(1.5)
            if self.var_sometimes.get()!=1:
                #self.img_i,self.bbs=iaa.Affine(rotate=rot_frac)(image=self.img_i,bounding_boxes=self.bbs)
                self.img_i,self.bbs=iaa.Affine(scale=(scale_frac1,scale_frac2))(image=self.img_i,bounding_boxes=self.bbs)
            else:
                if self.var_sometimes_Affine_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_Affine=float(self.var_sometimes_Affine_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_Affine=sometimes_frac
                else:
                    sometimes_frac_Affine=sometimes_frac
                #self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_Affine,iaa.Affine(rotate=rot_frac))(image=self.img_i,bounding_boxes=self.bbs)
                self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_Affine,iaa.Affine(scale=(scale_frac1,scale_frac2)))(image=self.img_i,bounding_boxes=self.bbs)
        if self.var_GrayScale.get()==1:
            try:
                GrayScale_frac=float(self.var_GrayScale_frac1.get())
            except:
                print('Exception using 1.0 for Maximum GrayScale_frac1')
                GrayScale_frac=float(1.0)
            if self.var_sometimes.get()!=1:
                self.img_i,self.bbs=iaa.Grayscale(alpha=(0.0,GrayScale_frac))(image=self.img_i,bounding_boxes=self.bbs)
            else:
                if self.var_sometimes_GrayScale_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_GrayScale=float(self.var_sometimes_GrayScale_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_GrayScale=sometimes_frac
                else:
                    sometimes_frac_GrayScale=sometimes_frac
                self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_GrayScale,iaa.Grayscale(alpha=(0.0,GrayScale_frac)))(image=self.img_i,bounding_boxes=self.bbs)

        if self.var_ColorTemp.get()==1:
            try:
                ColorTemp_frac1=float(self.var_ColorTemp_frac1.get())
            except:
                print('Exception using 1100 for ColorTemp_frac1')
                ColorTemp_frac1=float(1100)
            try:
                ColorTemp_frac2=float(self.var_ColorTemp_frac2.get())
            except:
                print('Exception using 10000 for ColorTemp_frac2')
                ColorTemp_frac2=float(10000)
            if self.var_sometimes.get()!=1:
                self.img_i,self.bbs=iaa.ChangeColorTemperature((ColorTemp_frac1,ColorTemp_frac2))(image=self.img_i,bounding_boxes=self.bbs)
            else:
                if self.var_sometimes_ColorTemp_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_ColorTemp=float(self.var_sometimes_ColorTemp_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_ColorTemp=sometimes_frac
                else:
                    sometimes_frac_ColorTemp=sometimes_frac
                self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_ColorTemp,iaa.ChangeColorTemperature((ColorTemp_frac1,ColorTemp_frac2)))(image=self.img_i,bounding_boxes=self.bbs)

        if self.var_GaussianBlur.get()==1:
            try:
                GaussianBlur_frac=float(self.var_GaussianBlur_frac1.get())
            except:
                print('Exception using 3.0 for Maximum GaussianBlur_frac1')
                GaussianBlur_frac=float(3.0)
            if self.var_sometimes.get()!=1:
                self.img_i,self.bbs=iaa.GaussianBlur(sigma=(0.0,GaussianBlur_frac))(image=self.img_i,bounding_boxes=self.bbs)
            else:
                if self.var_sometimes_GaussianBlur_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_GaussianBlur=float(self.var_sometimes_GaussianBlur_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_GaussianBlur=sometimes_frac
                else:
                    sometimes_frac_GaussianBlur=sometimes_frac
                self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_GaussianBlur,iaa.GaussianBlur(sigma=(0.0,GaussianBlur_frac)))(image=self.img_i,bounding_boxes=self.bbs)
        if self.var_AffineRotate.get()==1:
            try:
                str_rotations=self.var_AffineRotate_frac1.get()
                rotations=str_rotations.split(',')
                rotations=[int(w) for w in rotations]
            except:
                str_rotations="-45,-30,-15,-10,-5,5,10,15,30,45"
                print(f'Exception using {str_rotations} for min scale for Affine_frac')

                rotations=str_rotations.split(',')
                rotations=[int(w) for w in rotations]

            if self.var_sometimes.get()!=1:
                #self.img_i,self.bbs=iaa.Affine(rotate=rot_frac)(image=self.img_i,bounding_boxes=self.bbs)
                self.img_i,self.bbs=iaa.Affine(rotate=iap.Deterministic(np.random.choice(rotations)))(image=self.img_i,bounding_boxes=self.bbs)
            else:
                if self.var_sometimes_AffineRotate_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_AffineRotate=float(self.var_sometimes_AffineRotate_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_AffineRotate=sometimes_frac
                else:
                    sometimes_frac_AffineRotate=sometimes_frac
                #self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_Affine,iaa.Affine(rotate=rot_frac))(image=self.img_i,bounding_boxes=self.bbs)
                self.img_i,self.bbs=iaa.Sometimes(sometimes_frac_AffineRotate,iaa.Affine(rotate=iap.Deterministic(np.random.choice(rotations))))(image=self.img_i,bounding_boxes=self.bbs)
        if self.var_FakeImage.get()==1:
            if self.var_sometimes.get()!=1:
                self.img_i,self.bbs=self.put_fake_background(self.img_i,self.bbs)
            else:
                if self.var_sometimes_FakeImage_frac.get()!=self.var_sometimes_frac.get():
                    try:
                        sometimes_frac_FakeImage=float(self.var_sometimes_FakeImage_frac.get())
                    except:
                        print('Exception using sometimes fraction')
                        sometimes_frac_FakeImage=sometimes_frac
                else:
                    sometimes_frac_FakeImage=sometimes_frac
                random_i=np.random.random()
                if sometimes_frac_FakeImage>random_i:
                    self.img_i,self.bbs=self.put_fake_background(self.img_i,self.bbs)
                
    def close(self,event):
        self.root.destroy()

if __name__=='__main__':
    #path_JPEGImages=r"/media/steven/Elements/Drone_Videos/Combined_transporter9_only_upto_5_18_2022/JPEGImages"
    #path_Annotations=r"/media/steven/Elements/Drone_Videos/Combined_transporter9_only_upto_5_18_2022/Annotations"
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--path_JPEGImages",type=str,default='None',help='path JPEGImages')
    ap.add_argument("--path_Annotations",type=str,default='None',help='path Annotations')
    args = vars(ap.parse_args())
    path_JPEGImages=args['path_JPEGImages']
    path_Annotations=args['path_Annotations']
    get_default_settings()
    if path_JPEGImages=='None' or path_Annotations=='None':
        
        main_front=main_entry(root_tk)
        main_front.root.mainloop()
        root_tk=tk.Tk()
        myaug=IMGAug_JPGS_ANNOS(path_Annotations,path_JPEGImages)
        myaug.root.mainloop()
    else:
        myaug=IMGAug_JPGS_ANNOS(path_Annotations,path_JPEGImages)
        myaug.root.mainloop()
