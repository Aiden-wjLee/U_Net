from UNet import *

def filterDataset(folder, classes=None, mode='train'):    
    """
    initialize COCO api for instance annotations
    """
    annFile = '{}/{}.json'.format(folder, mode)
    coco = COCO(annFile)
    
    images = []
    if classes!=None:
        # iterate for each individual class in the list
        for className in classes:
            # get all images containing given categories
            catIds = coco.getCatIds(catNms=className)
            imgIds = coco.getImgIds(catIds=catIds)
            images += coco.loadImgs(imgIds)
    
    else:
        imgIds = coco.getImgIds()
        images = coco.loadImgs(imgIds)
    
    # Now, filter out the repeated images
    unique_images = []

    for i in range(len(images)): #len(images)는 100개. labeling 총 개수 (2x50)
        if images[i] not in unique_images:
            unique_images.append(images[i])
            
    random.shuffle(unique_images)
    dataset_size = len(unique_images)
    
    return unique_images, dataset_size, coco





#======================================
def getClassName(classID, cats):
    for i in range(len(cats)):
        if cats[i]['id']==classID:
            return cats[i]['name']
    return None

def getImage(imageObj, img_folder, input_image_size):
    # Read and normalize an image
    train_img = io.imread(img_folder + '/' + imageObj['file_name'])/255.0

    train_img=0.2126*train_img[:,:,0]+0.7152*train_img[:,:,1]+0.0722*train_img[:,:,2] ###변환 공식 사용
    ###train_img=cv2.resize(train_img,(256,256))

    ############print("getimage_imageobj", imageObj)
    ############print("train_img1: ", train_img.shape)
    # Resize
    train_img = cv2.resize(train_img, input_image_size)

    if (len(train_img.shape)==3 and train_img.shape[2]==3): # If it is a RGB 3 channel image
        return train_img
    else: # To handle a black and white image, increase dimensions to 3
        stacked_img = np.stack((train_img,)*3, axis=-1)
        return stacked_img
    
def getNormalMask(imageObj, classes, coco, catIds, input_image_size):
    '''
    normalMask용. (일반적인 경우 이것을 사용하면 됨)
        return : 
            train_mask : 각 클래스의 id에 따라 출력. 0(ground)값 이후 각 id값 출력
    '''
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    cats = coco.loadCats(catIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        className = getClassName(anns[a]['category_id'], cats)
        pixel_value = classes.index(className)+1 #background를 0으로 하기 위함 +1
        #new_mask=coco.annToMask(anns[a])*pixel_value
        new_mask = cv2.resize(coco.annToMask(anns[a])*pixel_value, (input_image_size[1],input_image_size[0]))    #(input_image_size))
        train_mask = np.maximum(new_mask, train_mask)

    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask  
    
def getBinaryMask(imageObj, coco, catIds, input_image_size):
    '''
    binarymask용. (dataGeneratorCoco를 부를 때의 classes에 속하는 class들을 모두 1, 나머지를 0으로 mask)
        returns : 
            train_mask : classess에 속하는 class들을 모두 1로 하여 리턴한다.
    '''
    annIds = coco.getAnnIds(imageObj['id'], catIds=catIds, iscrowd=None)
    anns = coco.loadAnns(annIds)
    train_mask = np.zeros(input_image_size)
    for a in range(len(anns)):
        new_mask = cv2.resize(coco.annToMask(anns[a]), input_image_size)
        
        #Threshold because resizing may cause extraneous values
        new_mask[new_mask >= 0.5] = 1
        new_mask[new_mask < 0.5] = 0

        train_mask = np.maximum(new_mask, train_mask)
    
    # Add extra dimension for parity with train_img size [X * X * 3]
    train_mask = train_mask.reshape(input_image_size[0], input_image_size[1], 1)
    return train_mask


def dataGeneratorCoco(images, classes, coco, folder, 
                      input_image_size=(2592,1944), batch_size=4, mode='train', mask_type='normal'):
    '''
    COCO data를 통해 data를 생성
        Returns:
            yield를 사용함으로써, 이 함수를 사용할 때에 하나씩 내보내게 된다. 
            visualizeGenerator함수에서 
            def visualizeGenerator(gen):
                img, mask = next(gen)
            와 같이 사용했다. 
            https://www.daleseo.com/python-yield/
    '''
    img_folder = '{}/{}'.format(folder, mode)
    dataset_size = len(images)
    catIds = coco.getCatIds(catNms=classes)
    
    c = 0
    while(True):
        img = np.zeros((batch_size, input_image_size[0], input_image_size[1], 3)).astype('float') 
        mask = np.zeros((batch_size, input_image_size[0], input_image_size[1], 1)).astype('float')
        for i in range(c, c+batch_size): #initially from 0 to batch_size, when c = 0
            imageObj = images[i]
            ############print("imageObj: ",imageObj)
            ### Retrieve Image ###
            train_img = getImage(imageObj, img_folder, input_image_size)
            
            ### Create Mask ###
            if mask_type=="binary":
                train_mask = getBinaryMask(imageObj, coco, catIds, input_image_size)
            
            elif mask_type=="normal":
                train_mask = getNormalMask(imageObj, classes, coco, catIds, input_image_size)   
                ##cv2.imshow("ab",train_mask.squeeze())
                ##cv2.imshow("as",train_img.squeeze())
                ##cv2.waitKey(0)      #확인용 cv2        
            
            # Add to respective batch sized arrays
            
            #train_img=np.transpose(train_img,(2,1,0))
            img[i-c] = train_img
            mask[i-c] = train_mask
        
        
        ##c+=batch_size
        ###if(c + batch_size >= dataset_size):
        ###    c=0
        ###    random.shuffle(images)
       
        c+=batch_size
        if(c + batch_size>= dataset_size):
            yield img, mask
            break
        
        yield img, mask







#======================================
def visualizeGenerator(gen):
    '''
    4장씩 visualize해주는 함수.
    왼쪽엔 labeling이, 오른쪽에는 원본 이미지가 출력된다.
    '''
    img, mask = next(gen)
    
    fig = plt.figure(figsize=(20, 10))
    outerGrid = gridspec.GridSpec(1, 2, wspace=0.1, hspace=0.1)
    
    for i in range(2):
        innerGrid = gridspec.GridSpecFromSubplotSpec(2, 2,
                        subplot_spec=outerGrid[i], wspace=0.05, hspace=0.05)

        for j in range(4):
            ax = plt.Subplot(fig, innerGrid[j])
            if(i==1):
                ax.imshow(img[j])
            else:
                ax.imshow(mask[j][:,:,0])
                
            ax.axis('off')
            fig.add_subplot(ax)        
    plt.show()


def get_one_hot_encoded_mask(mask_img):  
    """
    one_hot_encoder (class 3)
        help : if class increase, edit code
    """  
    image_batch=mask_img.shape[0]
    image_height=mask_img.shape[1]
    image_width=mask_img.shape[2]
    n_classes = 3
    y_img = np.squeeze(mask_img)#, axis=2)
    one_hot_mask = np.zeros((image_batch, image_height, image_width, n_classes))


    zero = (y_img == 0) 
    one = (y_img ==1)
    two = (y_img ==2)

    one_hot_mask[:, :, :, 0] = np.where(zero, 1, 0)  # if zero -> 1 not 0
    one_hot_mask[:, :, :, 1] = np.where(one, 1, 0)   # if two -> 1 not 0 
    one_hot_mask[:, :, :, 2] = np.where(two, 1, 0)
    '''
    back = (y_img == 0) 
    object = (y_img > 0)
    one_hot_mask[:, :, 0] = np.where(back, 1, 0)
    one_hot_mask[:, :, 1] = np.where(object, 1, 0)
    '''
    return one_hot_mask  

def save(ckpt_dir,net,optim,epoch):
    """
    네트워크 저장하기
    train을 마친 네트워크 저장 
    net : 네트워크 파라미터, optim  두개를 dict 형태로 저장
    """
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'net':net.state_dict(),'optim':optim.state_dict()},'%s/model_epoch%d.pth'%(ckpt_dir,epoch))

