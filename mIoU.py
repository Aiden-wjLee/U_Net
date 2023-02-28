from coco2dataset import *
#이전파일 : test.py
def mIoU(output, label, batch_size):
    """
    https://gaussian37.github.io/vision-segmentation-miou/#5-%ED%81%B4%EB%9E%98%EC%8A%A4-%EB%B3%84-iou-%EA%B3%84%EC%82%B0-1 참조
    mIou 측정, output : numpy, label : numpy
        Args:
            output_class : 클래스별로 0, 1, 2로 나타낸다.
    """
    n_class=3
    IoU=np.zeros(n_class)
    for i in range(batch_size):
        #class를 0, 1, 2로 표현
        label_class=label[i]
        output_class=output[i,1,:,:]*0+output[i,1,:,:]*1+output[i,2,:,:]*2

        #int 형 변경
        label_class=label_class.astype(np.int64)  
        output_class=output_class.astype(np.int64) 

        #행렬 -> 1D vector
        label_class_1D=label_class.reshape(-1)
        output_class_1D=output_class.reshape(-1)

        #bincount (빈도수 계산)
        label_cnt=np.bincount(label_class_1D)
        output_cnt=np.bincount(output_class_1D)

        #카테고리 행렬 생성. 3을 곱해주는 것은 ground, FC, IN 3개의 class로 이루어져 있기 때문
        category=label_class_1D*n_class+output_class_1D 
        
        #카테고리 행렬의 빈도수 계산
        category_cnt=np.bincount(category)
        category_cnt_2D=category_cnt.reshape(n_class,n_class)

        #IoU 계산 및 mIoU계산
        IoU_i=np.zeros(n_class)
        for j in range(n_class):
            union=np.sum(category_cnt_2D[j,:])+np.sum(category_cnt_2D[:,j])-category_cnt_2D[j,j]
            intersection=category_cnt_2D[j,j]
            IoU_i[j]=intersection/union
        IoU+=IoU_i
    print(IoU/batch_size)
    print("mIOU: ",np.average(IoU/batch_size))

def main():
    lr = 0.001
    batch_size = 10
    num_epoch = 5

    #folder = 'D:/OneDrive - Sogang/Sogang/22_winter/CV_study/learn_model/U_Net'
    folder = './'
    classes = ['FC','IN']
    mode = 'trainval_test'
    input_image_size_ = (32,32)
    mask_type = 'normal'
    images, dataset_size, coco = filterDataset(folder, classes,  mode)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #network & loss &optimizer
    net_test=UNet().to(device)
    net_test.load_state_dict(torch.load('./BEST/BEST35.pth'))

    fn_loss = nn.BCEWithLogitsLoss().to(device)

    optim = torch.optim.Adam(net_test.parameters(), lr = lr ) 

    start_time=time.time()
    #test
    with torch.no_grad(): # validation 이기 때문에 backpropa 진행 x, 학습된 네트워크가 정답과 얼마나 가까운지 loss만 계산
        net_test.eval() # 네트워크를 evaluation 용으로 선언
        loss_arr = []
        val_gen = dataGeneratorCoco(images, classes, coco, folder,
            input_image_size_, batch_size, mode, mask_type)

        for batch, (inputs,label) in enumerate(val_gen):
            print(batch) 
            # forward
            inputs=1/3*inputs[:,:,:,0]+1/3*inputs[:,:,:,1]+1/3*inputs[:,:,:,2]
            inputs_numpy=inputs
            inputs=np.expand_dims(inputs, axis=-1)
            #label=label/2   ###최대값 2->1

            inputs=np.transpose(inputs,(0,3,1,2))
            label=np.transpose(label,(0,3,1,2))
            inputs=torch.from_numpy(inputs)
            label=torch.from_numpy(label)

            label=label.float().to(device)
            inputs=inputs.float().to(device)

            output = net_test(inputs)

            output_numpy=output.cpu().numpy()
            #output_numpy=output_numpy/2
            
            output_numpy[output_numpy<0]=0; output_numpy[output_numpy>1]=1
            inputs_numpy[inputs_numpy<0]=0; inputs_numpy[inputs_numpy>1]=1
            output_numpy=np.around(output_numpy)
            
            in_out=output_numpy[:,1]+output_numpy[:,2]*(-1)+inputs_numpy

            mIoU(output_numpy, label.cpu().numpy(), batch_size)
            # loss 
            #loss = fn_loss(output,label)
            #loss_arr += [loss.item()]
    end_time=time.time()
    print("time: ",end_time-start_time)

if __name__=='__main__':
    main()