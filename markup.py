#OpenCVをインポート
import cv2


#動画から入力を開始（動画ファイルは各自で用意してください．）
cap = cv2.VideoCapture("test.mov")

while(True):
    
    #画像を取得
    _, img = cap.read()
    
    # ORB検出器を作る
    orb = cv2.ORB_create()
    
    # ORBでキーポイントを計算
    kp = orb.detect(img,None)
    
    # ORBで特徴記述子を計算
    kp, des = orb.compute(img, kp)
    
    # キーポイントの位置だけを描画、サイズと向きは描画しない
    img2 = cv2.drawKeypoints(img,kp,None,color=(0,0,255), flags=0)
    
    #映像として表示
    img3 = cv2.resize(img2, dsize=None, fx=0.4,fy=0.4)
    img3 = cv2.rotate(img3, cv2.ROTATE_180)

    cv2.imshow("ORB-SLAM", img3)
    
    #もし，エンターキーが押されたら，終了
    if cv2.waitKey(1)==13:
        break

cap.release()#カメラを開放
cv2.destroyAllWindows()#ウィンドウを破棄