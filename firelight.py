# firelight.py
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import open3d as o3d
from PIL import Image
import json
import mediapipe as mp

def add_flame_effect(edges, vitality):
    """
    エッジ部分に炎のような効果を追加します。
    """
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    flame_color = (255, 255, 255)  # 白色 (OpenCVでは0-255の範囲)
    mask = edges > 0
    edges_colored[mask] = flame_color

    # エッジを膨張させて太くする
    kernel = np.ones((3,3), np.uint8)
    edges_dilated = cv2.dilate(edges_colored, kernel, iterations=1)

    # グロー効果の作成
    glow = cv2.GaussianBlur(edges_dilated, (15,15), 0)

    # グローとエッジを重ね合わせて立体感を出す
    combined = cv2.addWeighted(glow, 0.6, edges_dilated, 0.4, 0)
    return combined

def create_depth_background(frame, gradient_top_color=(0, 0, 0), gradient_bottom_color=(255, 255, 255)):
    """
    フレームの背景に黒から白へのグラデーションを適用します。
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

def depth_to_point_cloud(depth_map, focal_length, cx, cy):
    """
    深度マップから3D点群を生成します。
    """
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    z = depth_map
    x = (i - cx) * z / focal_length
    y = (j - cy) * z / focal_length
    points = np.stack((x, y, z), axis=-1).reshape(-1, 3)
    # 無効なポイントを除去
    points = points[np.all(points != 0, axis=1)]
    return points

def draw_face_landmarks(image, results, mp_face_mesh, mp_drawing, mp_drawing_styles):
    """
    画像に顔のランドマークを描画します。
    
    Parameters:
        image (numpy.ndarray): 入力画像。
        results: MediapipeのFaceMeshの結果オブジェクト。
        mp_face_mesh: MediapipeのFaceMeshモジュール。
        mp_drawing: MediapipeのDrawingモジュール。
        mp_drawing_styles: MediapipeのDrawing Stylesモジュール。
    """
    if not results.multi_face_landmarks:
        return
    for face_landmarks in results.multi_face_landmarks:
        mp_drawing.draw_landmarks(
            image=image,
            landmark_list=face_landmarks,
            connections=mp_face_mesh.FACEMESH_TESSELATION,
            landmark_drawing_spec=None,
            connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())

def draw_pose_landmarks(image, pose_results, mp_pose, mp_drawing, mp_drawing_styles):
    """
    画像にPoseランドマークを描画します。
    
    Parameters:
        image (numpy.ndarray): 入力画像。
        pose_results: MediapipeのPoseの結果オブジェクト。
        mp_pose: MediapipeのPoseモジュール。
        mp_drawing: MediapipeのDrawingモジュール。
        mp_drawing_styles: MediapipeのDrawing Stylesモジュール。
    """
    if not pose_results.pose_landmarks:
        return
    mp_drawing.draw_landmarks(
        image=image,
        landmark_list=pose_results.pose_landmarks,
        connections=mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

def main():
    print("firelight.py スクリプトを開始します。")
    
    # Mediapipeの初期化
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False,
                        model_complexity=1,
                        enable_segmentation=False,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5)

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # MiDaSモデルのロード
    model_type = "MiDaS_small"  # "MiDaS" or "MiDaS_small"
    try:
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
    except RuntimeError as e:
        print(f"モデルのロードに失敗しました: {e}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    midas.to(device)
    midas.eval()

    # MiDaS_small 用のトランスフォームを手動で定義
    midas_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet の平均値
            std=[0.229, 0.224, 0.225]    # ImageNet の標準偏差
        )
    ])

    # カメラの初期化（Device ID 0）
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("カメラを開くことができませんでした。デバイスID 0を確認してください。")
        return

    print("点群生成とエッジ検出が起動しました。'q' キーを押すと終了します。's' キーを押すとオリジナル映像の表示を切り替えます。")

    # Open3Dのビジュアライザーのセットアップ
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='3D Point Cloud', width=800, height=600)
    point_cloud = o3d.geometry.PointCloud()
    vis.add_geometry(point_cloud)
    
    # 参考用の座標軸を追加
    coordinate_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])
    vis.add_geometry(coordinate_frame)

    # 背景色を黒に設定
    render_option = vis.get_render_option()
    render_option.background_color = np.asarray([0, 0, 0])  # 黒
    render_option.point_size = 1.0  # 点のサイズを調整

    # カメラキャリブレーションデータの読み込み
    try:
        with open("face_camera_calibration.json", "r") as f:
            calib_data = json.load(f)
        camera_matrix = np.array(calib_data["camera_matrix"], dtype=np.float32)
        dist_coeffs = np.array(calib_data["dist_coeffs"], dtype=np.float32).reshape(-1, 1)
        print("カメラキャリブレーションデータを読み込みました。")
    except FileNotFoundError:
        print("カメラキャリブレーションデータ 'face_camera_calibration.json' が見つかりません。")
        print("顔を用いたカメラキャリブレーションを実施してください。")
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        face_mesh.close()
        pose.close()
        return
    except json.JSONDecodeError:
        print("カメラキャリブレーションデータの読み込み中にエラーが発生しました。JSON形式を確認してください。")
        cap.release()
        cv2.destroyAllWindows()
        vis.destroy_window()
        face_mesh.close()
        pose.close()
        return

    # トラックバーのウィンドウを作成
    cv2.namedWindow('Enhanced Edge Flames with Depth')
    cv2.createTrackbar('Vitality', 'Enhanced Edge Flames with Depth', 50, 100, lambda x: None)

    # オリジナル映像（顔とポーズ）の表示フラグ
    show_original = False

    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("カメラからフレームを取得できませんでした。")
            break

        # Mediapipeで顔とポーズのランドマークを検出
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_mesh.process(frame_rgb)
        pose_results = pose.process(frame_rgb)

        # オリジナル映像の表示が有効な場合、ランドマークを描画
        if show_original:
            # フェイスランドマーク用のフレームコピー
            face_display = frame.copy()
            draw_face_landmarks(face_display, face_results, mp_face_mesh, mp_drawing, mp_drawing_styles)
            cv2.imshow('Face Landmarks', face_display)

            # ポーズランドマーク用のフレームコピー
            pose_display = frame.copy()
            draw_pose_landmarks(pose_display, pose_results, mp_pose, mp_drawing, mp_drawing_styles)
            cv2.imshow('Pose Landmarks', pose_display)
        else:
            # オリジナル映像のウィンドウが開いている場合、閉じる
            cv2.destroyWindow('Face Landmarks')
            cv2.destroyWindow('Pose Landmarks')

        # 歪み補正
        frame_undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, camera_matrix)

        # カメラ内部パラメータの設定
        focal_length = camera_matrix[0, 0]
        cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
        
        # フレームをグレースケールに変換
        gray = cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2GRAY)
        
        # エッジ検出
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        
        # トラックバーから生命力を取得
        vitality = cv2.getTrackbarPos('Vitality', 'Enhanced Edge Flames with Depth') / 100.0
        
        # 炎効果の追加
        flame_effect = add_flame_effect(edges, vitality)
        
        # 背景にグラデーションを適用（黒から白）
        background = create_depth_background(frame_undistorted)
        
        # 炎効果と背景を組み合わせる
        combined = combine_with_background(flame_effect, background)
        
        # オーバーレイとしてランドマークを描画（常に表示）
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                mp_drawing.draw_landmarks(
                    image=combined,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style())
        
        if pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                image=combined,
                landmark_list=pose_results.pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # 深度推定のためのトランスフォーム
        input_image = Image.fromarray(cv2.cvtColor(frame_undistorted, cv2.COLOR_BGR2RGB))
        input_batch = midas_transforms(input_image).unsqueeze(0).to(device)  # バッチ次元を追加
        
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_undistorted.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        
        depth_map = prediction.cpu().numpy()
        
        # 深度マップの正規化
        depth_min = depth_map.min()
        depth_max = depth_map.max()
        if depth_max - depth_min > 0:
            depth_map_normalized = (depth_map - depth_min) / (depth_max - depth_min)
            depth_map_normalized *= 10  # 深度をスケールアップ（必要に応じて調整）
        else:
            depth_map_normalized = np.zeros_like(depth_map)
        
        # 点群の生成
        points = depth_to_point_cloud(depth_map_normalized, focal_length, cx, cy)
        
        # Open3D用に点群データを作成
        if points.size > 0:
            point_cloud.points = o3d.utility.Vector3dVector(points)
            # 点群の色を白に設定（Open3Dは0-1の範囲）
            colors = np.ones((points.shape[0], 3))  # 白色
            point_cloud.colors = o3d.utility.Vector3dVector(colors)
            
            # ビジュアライザーを更新
            vis.update_geometry(point_cloud)
            vis.poll_events()
            vis.update_renderer()
        else:
            print("点群データがありません。")
        
        # エッジ効果の表示
        cv2.imshow('Enhanced Edge Flames with Depth', combined)
        
        # キーボード入力の処理
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("プログラムを終了します。")
            break
        elif key == ord('s'):
            show_original = not show_original
            if show_original:
                print("オリジナル映像の表示を有効にしました。")
            else:
                print("オリジナル映像の表示を無効にしました。")

    # クリーンアップ
    cap.release()
    cv2.destroyAllWindows()
    vis.destroy_window()
    face_mesh.close()
    pose.close()

if __name__ == "__main__":
    main()
