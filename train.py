from coco2dataset import *

import gc
gc.collect()
torch.cuda.empty_cache()
def main():
    #학습 관련 변수
    lr = 0.001
    batch_size = 5
    num_epoch = 35

    #image 정보
    #folder = 'D:/OneDrive - Sogang/Sogang/22_winter/CV_study/learn_model/U_Net' 
    folder = './' 
    classes = ['FC','IN']
    mode = 'trainval'
    input_image_size_ = (256,256)
    mask_type = 'normal'
    images, dataset_size, coco = filterDataset(folder, classes,  mode)

    #디바이스 및 네트워크 
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net=UNet().to(device)

    #loss 
    ##fn_loss = nn.BCEWithLogitsLoss().to(device) #original function
    fn_loss = nn.CrossEntropyLoss().to(device)
    #fn_loss = nn.BinaryCrossEntropy().to(device)

    #optimizer
    optim = torch.optim.Adam(net.parameters(), lr = lr ) 
    num_train = 50

    #variables
    num_train_for_epoch = np.ceil(num_train/batch_size) # np.ceil : 소수점 반올림

    # 기타 function 설정
    fn_tonumpy = lambda x : x.to('cpu').detach().numpy().transpose(0,2,3,1) # device 위에 올라간 텐서를 detach 한 뒤 numpy로 변환
    fn_denorm = lambda x, mean, std : (x * std) + mean 
    fn_classifier = lambda x :  1.0 * (x > 0.5)  # threshold 0.5 기준으로 indicator function으로 classifier 구현

    # Tensorbord
    #writer_train = SummaryWriter(log_dir=os.path.join(folder,'train')) ##train?
    '''
    # 네트워크 불러오기
    def load(ckpt_dir,net,optim):
        if not os.path.exists(ckpt_dir): # 저장된 네트워크가 없다면 인풋을 그대로 반환
            epoch = 0
            return net, optim, epoch
        
        ckpt_lst = os.listdir(ckpt_dir) # ckpt_dir 아래 있는 모든 파일 리스트를 받아온다
        ckpt_lst.sort(key = lambda f : int(''.join(filter(str,isdigit,f))))

        dict_model = torch.load('%s/%s' % (ckpt_dir,ckpt_lst[-1]))

        net.load_state_dict(dict_model['net'])
        optim.load_state_dict(dict_model['optim'])
        epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

        return net,optim,epoch
    '''
    best_loss=float('inf')

    start_epoch = 0
    for epoch in range(start_epoch+1,num_epoch +1):
        net.train()
        loss_arr = []
        
        val_gen = dataGeneratorCoco(images, classes, coco, folder,
        input_image_size_, batch_size, mode, mask_type)

        for batch, (inputs,label) in enumerate(val_gen): # 1은 뭐니 > index start point
            inputs=1/3*inputs[:,:,:,0]+1/3*inputs[:,:,:,1]+1/3*inputs[:,:,:,2]
            inputs=np.expand_dims(inputs, axis=-1)

            inputs=np.transpose(inputs,(0,3,1,2))  
            label=np.transpose(label,(0,3,1,2))   ##if using BCE loss
            inputs=torch.from_numpy(inputs)
            label=torch.from_numpy(label)

            label=label.long().to(device) # multi class classification 을 한 long type변경 
            inputs=inputs.float().to(device)

            output = net(inputs) 
            
            # backward
            optim.zero_grad()  # gradient 초기화
            #output=np.transpose(output.cpu(),(0,2,3,1))#.contiguous()

            loss = fn_loss(output, label.squeeze())  # output과 label 사이의 loss 계산  # 5x3x256x256과 5x256x256
            loss.backward() # gradient backpropagation
            optim.step() # backpropa 된 gradient를 이용해서 각 layer의 parameters update
            
            # save loss
            loss_arr += [loss.item()]

            # tensorboard에 결과값들 저정하기
            label = fn_tonumpy(label)
            inputs = fn_tonumpy(fn_denorm(inputs,0.5,0.5))
            output = fn_tonumpy(fn_classifier(output))
            if batch%5==0:
                print("batch[ {} / {} ], epoch: [ {} / {} ], loss: {}".format(batch, int(num_train/batch_size), epoch, num_epoch,np.mean(loss_arr)))

        # epoch이 끝날때 마다 비교 후 네트워크 저장
        #save(ckpt_dir=os.path.join(folder,'pth'), net = net, optim = optim, epoch = epoch)
        if np.mean(loss_arr) <= best_loss:
            best_loss=np.mean(loss_arr)
            torch.save(net.state_dict(), "./BEST/BEST{}.pth".format(epoch))

if __name__=='__main__':
    main()