#数据增强（代码详见./data/datacleaning_augmentation.py）
1.将训练集中H或W大于500的图像进行双线性插值下采样，以更接近test集的图像尺度。
2.通过计算test和train图像的SSIM，挑选出最接近的图像，进行4倍扩充后填补入训练集。
3.增大patchsize可有效提点，算法中取528
4.增大batchsize可有效提点，显存有限，训练取16

#loss
MSEloss

#模型
在IMDN基础上进行改进，加入注意力机制，使用prelu替换lrelu,为弥补带来的计算量增加，减少一层block

#训练
sudo python3 ./codes/train.py

#测试
test时，高低分辨率图像都需要
sudo python3 ./codes/test.py

#线上成绩
best score: psnr=8.619773372258521，ssim=0.29088628652944015,time(同一模型存在波动）=0.09999999999999999~0.05217391304347826




