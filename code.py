import mindspore 
from mindspore import dataset as ds
from mindspore.dataset import transforms
from mindspore import nn 
import numpy as np 
import matplotlib.pyplot as plt 
from netCDF4 import Dataset
import os 


mindspore.run_check()


# 按月份分别建立模型
# 按预测长度分别建立模型
# 目标月份和预测长度是循环中重要的变量
target_mons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]    # 目标月份
lead_mons = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]      # 预测长度

# 月份名称
mon_name = ['JFM','FMA','MAM','AMJ','MJJ','JJA','JAS','ASO','SON','OND','NDJ','DJF']

# 数据维度
xdim = 72
ydim = 24
zdim = 6

# CNN中其他相关参数
batch_size = 400        # Batch Size
conv_drop = 0.5         # drop rate of the convolutional layer
hidd_drop = 0.5         # drop rate of the hidden layer
epoch = 50              # Epoch

for lead_mon in lead_mons:
    for target_mon in target_mons:
        print(f'lead_mon={lead_mon}, target_mon={target_mon}')

        tg_mn = int(target_mon - 1)
        # Q: 这里为什么要用23 - lead_mon + tg_mn呢?
        # A: 因为图片数据比标签数据早了两年, 所以要加上23, 其他操作也是为了时间的匹配
        ld_mn1 = int(23 - lead_mon + tg_mn)
        ld_mn2 = int(23 - lead_mon + tg_mn + 3)



        # ------------------------------------------------------读取数据----------------------------------------------------------------
        # 读取CMIP5数据集
        samfile = 'dataset/CMIP5/CMIP5.input.36mn.1861_2001.nc'
        labfile = 'dataset/CMIP5/CMIP5.label.nino34.12mn_3mv.1863_2003.nc'
        sample_size = 2961      # Training data size of training set
        inp1 = Dataset(samfile) # 图像数据
        inp2 = Dataset(labfile) # 标签数据
        inpv1 = np.zeros((sample_size, zdim, ydim, xdim))
        inpv1[:, 0: 3, :, :] = inp1.variables['sst1'][0: sample_size, ld_mn1: ld_mn2, :, :]
        inpv1[:, 3: 6, :, :] = inp1.variables['t300'][0: sample_size, ld_mn1: ld_mn2, :, :]
        inpv2 = inp2.variables['pr'][0: sample_size, tg_mn, 0]
        print('CPMI5:')
        print(inpv1.shape, inpv2.shape)

        # 定义CPIM5数据集
        class CPIM5_Dataset:
            """  
            CPIM5数据集
            """
            def __init__(self, inpv1, inpv2):
                self._data = inpv1
                self._label = inpv2 
            
            def __getitem__(self, index):
                return self._data[index], self._label[index]
            
            def __len__(self):
                return len(self._data)

        def datapipe(dataset, batch_size):
            image_transforms = transforms.TypeCast(mindspore.float32)
            label_transforms = transforms.TypeCast(mindspore.float32)
            dataset = dataset.map(image_transforms, 'image')
            dataset = dataset.map(label_transforms, 'label')
            dataset = dataset.batch(batch_size)
            return dataset

        dataset_cpim5 = CPIM5_Dataset(inpv1, inpv2)
        dataset_cpim5 = ds.GeneratorDataset(source=dataset_cpim5, column_names=['image', 'label'], shuffle=True)
        dataset_cpim5 = datapipe(dataset_cpim5, batch_size)

        for image, label in dataset_cpim5.create_tuple_iterator():
            print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
            print(f"Shape of label: {label.shape} {label.dtype}")
            break
        print('CPIM5 done!\n')


        # 读取SODA数据集
        samfile = './dataset/SODA/SODA.input.36mn.1871_1970.nc'
        labfile = './dataset/SODA/SODA.label.nino34.12mn_3mv.1873_1972.nc'
        sample_size = 100

        inp1 = Dataset(samfile)
        inp2 = Dataset(labfile)
        inpv1 = np.zeros((sample_size, zdim, ydim, xdim))
        inpv1[:, 0: 3, :, :] = inp1.variables['sst'][0: sample_size, ld_mn1: ld_mn2, :, :]
        inpv1[:, 3: 6, :, :] = inp1.variables['t300'][0: sample_size, ld_mn1: ld_mn2, :, :]
        inpv2 = inp2.variables['pr'][0: sample_size, tg_mn, 0]
        print('SODA:')
        print(inpv1.shape, inpv2.shape)

        # 定义SODA数据集
        class SODA_Dataset:
            """  
            SODA数据集
            """
            def __init__(self, inpv1, inpv2):
                self._data = inpv1
                self._label = inpv2 
            
            def __getitem__(self, index):
                return self._data[index], self._label[index]
            
            def __len__(self):
                return len(self._data)

        def datapipe(dataset, batch_size):
            image_transforms = transforms.TypeCast(mindspore.float32)
            label_transforms = transforms.TypeCast(mindspore.float32)
            dataset = dataset.map(image_transforms, 'image')
            dataset = dataset.map(label_transforms, 'label')
            dataset = dataset.batch(batch_size)
            return dataset

        dataset_soda = SODA_Dataset(inpv1, inpv2)
        dataset_soda = ds.GeneratorDataset(source=dataset_soda, column_names=['image', 'label'], shuffle=True)
        dataset_soda = datapipe(dataset_soda, batch_size)

        for image, label in dataset_soda.create_tuple_iterator():
            print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
            print(f"Shape of label: {label.shape} {label.dtype}")
            break
        print('SODA done!\n')


        # 读取GODAS数据
        samfile = 'dataset/GODAS/GODAS.input.36mn.1980_2015.nc'
        labfile = 'dataset/GODAS/GODAS.label.12mn_3mv.1982_2017.nc'
        sample_size = 36          # Train size of GODAS

        inp1 = Dataset(samfile) # 图像数据
        inp2 = Dataset(labfile) # 标签数据
        inpv1 = np.zeros((sample_size, zdim, ydim, xdim))
        inpv1[:, 0: 3, :, :] = inp1.variables['sst'][0: sample_size, ld_mn1: ld_mn2, :, :]
        inpv1[:, 3: 6, :, :] = inp1.variables['t300'][0: sample_size, ld_mn1: ld_mn2, :, :]
        inpv2 = inp2.variables['pr'][0: sample_size, tg_mn, 0]
        print('GODAS')
        print(inpv1.shape, inpv2.shape)

        # 定义GODAS数据集
        class GODAS_Dataset:
            """  
            GODAS数据集
            """
            def __init__(self, inpv1, inpv2):
                self._data = inpv1
                self._label = inpv2 
            
            def __getitem__(self, index):
                return self._data[index], self._label[index]
            
            def __len__(self):
                return len(self._data)

        def datapipe(dataset, batch_size):
            image_transforms = transforms.TypeCast(mindspore.float32)
            label_transforms = transforms.TypeCast(mindspore.float32)
            dataset = dataset.map(image_transforms, 'image')
            dataset = dataset.map(label_transforms, 'label')
            dataset = dataset.batch(batch_size)
            return dataset

        dataset_godas = GODAS_Dataset(inpv1, inpv2)
        dataset_godas = ds.GeneratorDataset(source=dataset_godas, column_names=['image', 'label'], shuffle=False)
        dataset_godas = datapipe(dataset_godas, batch_size)

        for image, label in dataset_godas.create_tuple_iterator():
            print(f"Shape of image [N, C, H, W]: {image.shape} {image.dtype}")
            print(f"Shape of label: {label.shape} {label.dtype}")
            break
        print('GODAS done!')


        # -----------------------------------------------模型结构--------------------------------------------------------------------
        
        # 初始化卷积层与BatchNorm的参数
        class CNN(nn.Cell):
            """  
            卷积神经网络
            """
            def __init__(self, num_convf, num_hiddf):
                super(CNN, self).__init__()
                self.conv1 = nn.Conv2d(in_channels=6, out_channels=num_convf, kernel_size=(8, 4))
                self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
                self.conv2 = nn.Conv2d(in_channels=num_convf, out_channels=num_convf, kernel_size=(4, 2))
                self.conv3 = nn.Conv2d(in_channels=num_convf, out_channels=num_convf, kernel_size=(4, 2))
                self.fc1 = nn.Dense(in_channels=108 * num_convf, out_channels=num_hiddf)
                self.fc2 = nn.Dense(in_channels=num_hiddf, out_channels=1)
                self.tanh = nn.Tanh()
                self.dropout1 = nn.Dropout(p=conv_drop)
                self.dropout2 = nn.Dropout(p=hidd_drop)
                self.flatten = nn.Flatten()

            def construct(self, x):
                # conv1
                x = self.tanh(self.conv1(x))
                x = self.pool(x)
                x = self.dropout1(x)
                # conv2
                x = self.tanh(self.conv2(x))
                x = self.pool(x)
                x = self.dropout1(x)
                # conv3
                x = self.tanh(self.conv3(x))
                x = self.flatten(x)
                x = self.dropout1(x)
                # fc1
                x = self.tanh(self.fc1(x))
                x = self.dropout2(x)
                # fc2
                x = self.fc2(x)
                return x 
            

        # ----------------------------------------训练和预测(CMIM5预训练、SODA迁移学习、GODAS验证)-------------------------------------------

        # Define function for train 
        def train(model, dataset):
            # Instantiate loss function and optimizer
            loss_fn = nn.MSELoss()
            optimizer = nn.RMSProp(params=model.trainable_params(), learning_rate=5e-3, decay=0.9)

            # 1. Define forward function
            def forward_fn(data, label):
                pred = model(data)
                loss = loss_fn(pred, label)
                return loss, pred

            # 2. Get gradient function
            grad_fn = mindspore.value_and_grad(forward_fn, None, optimizer.parameters, has_aux=True)

            # 3. Define function of one-step training
            def train_step(data, label):
                (loss, _), grads = grad_fn(data, label)
                optimizer(grads)
                return loss 
            
            # train
            size = dataset.get_dataset_size()
            model.set_train()
            for batch, (data, label) in enumerate(dataset.create_tuple_iterator()):
                loss = train_step(data, label)
                if batch + 1 == size:
                    loss, current = loss.asnumpy(), batch 
                    print(f"loss: {loss:>7f}")


        def test(model, dataset):
            # 模型预测
            model.set_train(False)
            for data, label in dataset.create_tuple_iterator():
                pred = model(data)
            return pred.asnumpy().flatten(), label.asnumpy().flatten()

        model_1 = CNN(30, 30)
        model_2 = CNN(30, 50)
        model_3 = CNN(50, 30)
        model_4 = CNN(50, 50)
        # 模型训练+迁移学习+预测
        for model in [model_1, model_2, model_3, model_4]:
            for i in range(epoch):
                print(f"Epoch {i + 1}\n-----------------")
                train(model, dataset_cpim5)
            print('开始迁移学习...')
            for i in range(epoch):
                print(f"Epoch {i + 1}\n-----------------")
                train(model, dataset_soda)
            print('one model is done!')
            print('\n')
            
        # 保存模型
        os.makedirs(f'./model/lead_mon={lead_mon}/target_mon={target_mon}', exist_ok=True)
        mindspore.save_checkpoint(model_1, f'./model/lead_mon={lead_mon}/target_mon={target_mon}/model_1.ckpt')
        mindspore.save_checkpoint(model_2, f'./model/lead_mon={lead_mon}/target_mon={target_mon}/model_2.ckpt')
        mindspore.save_checkpoint(model_3, f'./model/lead_mon={lead_mon}/target_mon={target_mon}/model_3.ckpt')
        mindspore.save_checkpoint(model_4, f'./model/lead_mon={lead_mon}/target_mon={target_mon}/model_4.ckpt')

        # 预测
        pred_1, label = test(model_1, dataset_godas)
        pred_2, label = test(model_2, dataset_godas)
        pred_3, label = test(model_3, dataset_godas)
        pred_4, label = test(model_4, dataset_godas)
        pred = np.array([pred_1, pred_2, pred_3, pred_4])
        np.save(f'./preds/lead_mon={lead_mon}_target_mon={target_mon}_preds.npy', pred)
        pred_label = np.array([pred.mean(axis=0).flatten(), label])
        np.save(f'./pred_label/lead_mon={lead_mon}_target_mon={target_mon}.npy', pred_label)
        print(np.corrcoef(pred.mean(axis=0), label)[0][1])
        plt.scatter(pred.mean(axis=0), label)
        plt.show()

        # 删除不必要的变量以减少内存
        del dataset_cpim5
        del dataset_soda
        del dataset_godas
        del model_1
        del model_2
        del model_3
        del model_4