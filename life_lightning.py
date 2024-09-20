import cv2
import numpy as np
import time

def map_vitality_to_color(vitality):
    """
    生命力の強さに応じて色をマッピングします。
    生命力が高いほど黄色、低いほど青色になります。
    
    Parameters:
    - vitality (float): 0.0（低）から1.0（高）の値。
    
    Returns:
    - tuple: BGR形式の色。
    """
    # 黄色（高生命力）と青色（低生命力）のグラデーション
    blue = int(255 * (1 - vitality))
    green = int(255 * vitality)
    red = 255  # 常に最大値
    return (blue, green, red)

def add_flame_effect(edges, vitality):
    """
    エッジ部分に炎のような効果を追加します。
    
    Parameters:
    - edges (numpy.ndarray): エッジ検出結果の画像。
    - vitality (float): 生命力の強さ（0.0〜1.0）。
    
    Returns:
    - numpy.ndarray: 炎効果が追加されたカラー画像。
    """
    # エッジをカラー画像に変換
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # 生命力に応じた色を取得
    flame_color = map_vitality_to_color(vitality)
    
    # エッジ部分に色を適用
    mask = edges > 0
    edges_colored[mask] = flame_color
    
    # エッジを膨張させて太くする
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges_colored, kernel, iterations=1)
    
    # グロー効果の作成
    glow = cv2.GaussianBlur(edges_dilated, (15,15), 0)
    
    # グローとエッジを重ね合わせて立体感を出す
    combined = cv2.addWeighted(glow, 0.6, edges_colored, 0.4, 0)
    
    return combined

def create_depth_background(frame, gradient_top_color=(255, 0, 0), gradient_bottom_color=(0, 215, 255)):
    """
    フレームの背景に寒色系統から黄金色へのグラデーションを適用します。
    
    Parameters:
    - frame (numpy.ndarray): 元のフレーム画像。
    - gradient_top_color (tuple): グラデーションの上部の色（BGR）。
    - gradient_bottom_color (tuple): グラデーションの下部の色（BGR）。
    
    Returns:
    - numpy.ndarray: グラデーションが適用された背景画像。
    """
    height, width = frame.shape[:2]
    gradient = np.zeros((height, width, 3), dtype=np.uint8)
    
    for y in range(height):
        alpha = y / height
        color = [
            int(gradient_top_color[i] * (1 - alpha) + gradient_bottom_color[i] * alpha)
            for i in range(3)
        ]
        gradient[y, :] = color
    
    return gradient

def combine_with_background(flame_effect, background):
    """
    炎効果と背景を組み合わせます。
    
    Parameters:
    - flame_effect (numpy.ndarray): 炎効果が適用された画像。
    - background (numpy.ndarray): グラデーション背景画像。
    
    Returns:
    - numpy.ndarray: 背景と炎効果が組み合わさった最終画像。
    """
    # エッジ部分のマスクを作成
    edges_gray = cv2.cvtColor(flame_effect, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(edges_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    
    # 背景部分をマスクして取り出す
    background_part = cv2.bitwise_and(background, background, mask=mask_inv)
    
    # 炎効果部分をマスクして取り出す
    flame_part = cv2.bitwise_and(flame_effect, flame_effect, mask=mask)
    
    # 背景と炎効果を合成
    combined = cv2.add(background_part, flame_part)
    
    return combined

def nothing(x):
    pass

def main():
    # カメラの初期化（Device ID 0）
    cap = cv2.VideoCapture(0)
    
    # カメラが正しく開けたか確認
    if not cap.isOpened():
        print("カメラを開くことができませんでした。デバイスID 0を確認してください。")
        return
    
    print("エッジ検出と炎エフェクトが起動しました。'q' キーを押すと終了します。")
    
    # トラックバーのウィンドウを作成
    cv2.namedWindow('Enhanced Edge Flames with Depth')
    cv2.createTrackbar('Vitality', 'Enhanced Edge Flames with Depth', 50, 100, nothing)
    
    # 炎エフェクトの動的変化用タイマー
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
        
        # トラックバーから生命力を取得
        vitality = cv2.getTrackbarPos('Vitality', 'Enhanced Edge Flames with Depth') / 100.0
        
        # エッジ部分に炎の効果を追加
        flame_effect = add_flame_effect(edges, vitality)
        
        # 背景にグラデーションを適用
        background = create_depth_background(frame)
        
        # 炎効果と背景を組み合わせる
        combined = combine_with_background(flame_effect, background)
        
        # 結果を表示
        cv2.imshow('Enhanced Edge Flames with Depth', combined)
        
        # 'q' キーが押されたらループを抜ける
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("エッジ検出と炎エフェクトを終了します。")
            break
    
    # カメラを解放し、全てのOpenCVウィンドウを閉じる
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
