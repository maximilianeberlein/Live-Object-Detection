import cv2
import os
import re
import numpy as np

class_list = []
color_dic = dict()
flag = 0

def color_gen():
    '''
    Generate a new color. As first color generates (0, 255, 0)
    '''
    global flag
  
    if flag == 0:
        color = (0, 255, 0)
        flag += 1
    else:
        np.random.seed()
        color = tuple(255 * np.random.rand(3))
    return color

def show(class_name, download_dir, label_dir, images_files, index, args):
    '''
    Show the images with the labeled boxes.

    :param class_name: self explanatory
    :param download_dir: folder that contains the images
    :param label_dir: folder that contains the labels
    :param index: self explanatory
    :return: index
    '''
 
    if index > len(images_files)-1:
        index=0
    elif index<0:
        index=len(images_files)-1

    global class_list, color_dic

    total_images = len(images_files)

    current_image_path = images_files[index]
    
    img = cv2.imread(current_image_path)
    


    file_name = str(current_image_path.split('/')[-1].split('.')[0]) + '.txt'
    file_path = os.path.join(label_dir, file_name)
    f = open(file_path, 'r')

    window_name = "Visualizer: {}/{}".format(index+1, total_images)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    width = 500
    try:
        height = int((img.shape[0] * width) / img.shape[1])
    except Exception as e:
        print(e)
        print("[ERROR] Corrupted image => ", current_image_path)
        return  index

    cv2.resizeWindow(window_name, width, height)

    for line in f:        
        # each row in a file is class_name, XMin, YMix, XMax, YMax
        match_class_name = re.compile('^[a-zA-Z]+(\s+[a-zA-Z]+)*').match(line)
        
        if match_class_name==None and args.classes!=None:
            class_name = args.classes[int(line.split(' ')[0])]

        
        elif match_class_name!=None:
            class_name = line[:match_class_name.span()[1]]
        
        else:
            print("[ERROR] if you are using labels format make sure u provide classes list with arguments")
            exit(1)

        ax = line.split(' ')[1:]
	# opencv top left bottom right

        if class_name not in class_list:
            class_list.append(class_name)
            color = color_gen()     
            color_dic[class_name] = color  

        font = cv2.FONT_HERSHEY_SIMPLEX
        r ,g, b = color_dic[class_name]
        
        
        if args.yoloLabelStyle:
            w = int(float(ax[2]) * img.shape[1])
            h = int(float(ax[3]) * img.shape[0])

            x = int((float(ax[0]) * img.shape[1])-(w/2.0))
            y = int((float(ax[1]) * img.shape[0])-(h/2.0))


            cv2.putText(img,class_name,(x+5,y-7), font, 0.8,(b, g, r), 2,cv2.LINE_AA)
            cv2.rectangle(img, (x, y), (x+w, y+h), (b, g, r), 3)


        else:
            cv2.putText(img,class_name,(int(float(ax[0]))+5,int(float(ax[1]))-7), font, 0.8,(b, g, r), 2,cv2.LINE_AA)
            cv2.rectangle(img, (int(float(ax[-2])), int(float(ax[-1]))),
                        (int(float(ax[-4])),
                        int(float(ax[-3]))), (b, g, r), 3)

    cv2.imshow(window_name, img)

    return index
