import cv2
import numpy as np
import time

def main():
    # カメラの初期化（Device ID 0）
    cap = cv2.VideoCapture(0)
    
    # カメラが正しく開けたか確認
    if not cap.isOpened():
        print("カメラを開くことができませんでした。デバイスID 0を確認してください。")
        return
    
    print("エッジ検出カメラが起動しました。'q' キーを押すと終了します。")
    
    # 動的なエフェクト用の変数
    start_time = time.time()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("カメラからフレームを取得できませんでした。")
            break
        
        # フレームをグレースケールに変換
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # ガウシアンブラーを適用してノイズを低減
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Cannyエッジ検出を適用
        edges = cv2.Canny(blurred, 50, 150)
        
        # エッジを膨張させて太くする
        kernel = np.ones((3,3), np.uint8)
        edges_dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # グロー効果の作成
        glow = cv2.GaussianBlur(edges_dilated, (15,15), 0)
        
        # カラーマップの適用（例: ホットカラーマップ）
        glow_colored = cv2.applyColorMap(glow, cv2.COLORMAP_HOT)
        
        # 動的なカラー効果のためのフェードイン・フェードアウト
        elapsed_time = time.time() - start_time
        intensity = (np.sin(elapsed_time * 2) + 1) / 2  # 0から1の範囲
        glow_colored = cv2.convertScaleAbs(glow_colored, alpha=intensity)
        
        # エッジ部分にグロー効果を重ねる
        edges_colored = cv2.applyColorMap(edges_dilated, cv2.COLORMAP_HOT)
        combined = cv2.addWeighted(glow_colored, 0.6, edges_colored, 0.4, 0)
        
        # エッジ部分のみ表示
        cv2.imshow('Enhanced Edge Detection', combined)
        
        # 'q' キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("エッジ検出カメラを終了します。")
            break
    
    # カメラを解放し、全てのOpenCVウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
