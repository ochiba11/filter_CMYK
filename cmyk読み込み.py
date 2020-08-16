from PIL import Image,ImageFilter
import numpy as np
import cv2

#パス設定
input_path = "input\\"
input_name = "cmyk_moji02.jpg"
output_path = "output\\"
output_name = "cmyk_flt.jpg"
pmt_name = "pmt.txt"

#画像読み込み
img = Image.open(input_path+input_name)
print("読み込みinfo",img)

#4チャネルに分割する
c,m,y,k = img.split()
np_c = np.array(c)
np_m = np.array(m)
np_y = np.array(y)
np_k = np.array(k)

#カーネル用意
pmt = np.loadtxt(input_path+pmt_name,dtype="float32")
pmt_sum = pmt.sum()
kernel = pmt / pmt_sum

#フィルタ処理
flt_c = cv2.filter2D(np_c,-1,kernel)
flt_m = cv2.filter2D(np_m,-1,kernel)
flt_y = cv2.filter2D(np_y,-1,kernel)
flt_k = cv2.filter2D(np_k,-1,kernel)

#チャネルの結合
#次元の増加
up_dim_c = flt_c[:,:,np.newaxis]
up_dim_m = flt_m[:,:,np.newaxis]
up_dim_y = flt_y[:,:,np.newaxis]
up_dim_k = flt_k[:,:,np.newaxis]
#結合
np_union = np.concatenate((up_dim_c,up_dim_m,up_dim_y,up_dim_k),-1)

#pilImage化
pilImg = Image.fromarray(np.uint8(np_union),"CMYK")
print("変換info",pilImg)
pilImg.save(output_path+output_name)