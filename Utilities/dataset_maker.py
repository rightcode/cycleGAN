from PIL import Image  ### 画像処理ライブラリPillow をインポート
count = 0
start = 20000
end = 30000
man = "1"
woman = "-1"
with open("./drive/My Drive/list_attr_celeba.txt","r") as f:    ### 属性ファイルを開く
     for i in range(end):   ### 全部で202,599枚処理する
         
         line = f.readline()   ### 1行データ読み込み
         if i < start:continue
         line = line.split()  ### データを分割
         count = count+1
         
         if line[21] == woman:
          try:
             image = Image.open("./drive/My Drive/face/tmp/1/"+line[0])
         
             image.save("./drive/My Drive/woman/sub/"+line[0])
          except:
            1