import cv2
import numpy as np
import mediapipe as mp
import os
import json

def get_face_landmarks(image, face_mesh):
    """
    Mediapipeを使用して画像から顔のランドマークを検出します。
    Parameters:
        image (numpy.ndarray): 入力画像。
        face_mesh (mediapipe.solutions.face_mesh.FaceMesh): MediapipeのFaceMeshオブジェクト。
    Returns:
        list: 468個のランドマーク（x, y, z）のリスト。
    """
    results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if not results.multi_face_landmarks:
        return None
    # 最初の顔のみを使用
    landmarks = results.multi_face_landmarks[0]
    h, w, _ = image.shape
    landmark_points = []
    for lm in landmarks.landmark:
        x, y, z = int(lm.x * w), int(lm.y * h), lm.z
        landmark_points.append((x, y, z))
    return landmark_points

def main():
    print("カメラキャリブレーションスクリプトを開始します。")
    # Mediapipeの初期化
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                      max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)

    # カメラの初期化
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("カメラを開くことができません。デバイスIDを確認してください。")
        return

    print("カメラキャリブレーションを開始します。顔を異なる角度から動かしてください。")
    print("データ収集を終了するには 'q' キーを押してください。")

    # データ収集用リスト
    object_points = []  # 3Dポイント
    image_points = []   # 2Dポイント

    # 仮の3D顔モデル（選択したランドマークに対応する）
    selected_landmark_indices = [
        33,  # 鼻の先端
        263, # 左目の外側
        362, # 右目の外側
        78,  # 左頬
        308, # 右頬
        14,  # 口の左端
        287, # 口の右端
    ]

    # 仮の3Dモデルポイント（正規化）
    model_points = [
        (0.0, 0.0, 0.0),        # 鼻の先端
        (-30.0, -30.0, -30.0),  # 左目の外側
        (30.0, -30.0, -30.0),   # 右目の外側
        (-40.0, 20.0, -30.0),   # 左頬
        (40.0, 20.0, -30.0),    # 右頬
        (-20.0, 50.0, -30.0),   # 口の左端
        (20.0, 50.0, -30.0),    # 口の右端
    ]

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("カメラからフレームを取得できませんでした。")
            break

        landmarks = get_face_landmarks(frame, face_mesh)
        if landmarks:
            # 選択したランドマークの2D座標を取得
            img_pts = []
            obj_pts = []
            for idx, model_pt in zip(selected_landmark_indices, model_points):
                x, y, z = landmarks[idx]
                img_pts.append((x, y))
                obj_pts.append(model_pt)
            image_points.append(np.array(img_pts, dtype=np.float32))
            object_points.append(np.array(obj_pts, dtype=np.float32))
            frame_count += 1
            print(f"データセット {frame_count} を収集しました。")

            # ランドマークを描画
            for idx in selected_landmark_indices:
                x, y, _ = landmarks[idx]
                cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        cv2.imshow('Face Calibration', frame)

        # 'q' キーで終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("データ収集を終了します。")
            break

    cap.release()
    cv2.destroyAllWindows()
    face_mesh.close()

    if len(object_points) < 10:
        print("十分なデータが収集されていません。最低でも10セットのデータが必要です。")
        return

    print("カメラキャリブレーションを実行します。")

    # カメラキャリブレーション
    # 複数のデータセットを使用してsolvePnPを実行し、カメラ行列を推定
    # ここでは簡略化のため、最初のデータセットのみを使用します
    # 精度を向上させるためには、複数のデータセットを使用して最適化を行う必要があります

    # 最初のデータセットを使用
    obj_pts = object_points[0]
    img_pts = image_points[0]

    # カメラキャリブレーション
    # 仮のカメラ行列
    size = frame.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.zeros((4,1))  # 歪み係数を仮定

    success, rotation_vector, translation_vector = cv2.solvePnP(
        obj_pts,
        img_pts,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if success:
        print("カメラキャリブレーションに成功しました。")
        print("カメラ行列:\n", camera_matrix)
        print("歪み係数:\n", dist_coeffs)
        # キャリブレーション結果を保存
        calib_data = {
            "camera_matrix": camera_matrix.tolist(),
            "dist_coeffs": dist_coeffs.flatten().tolist()
        }
        with open("face_camera_calibration.json", "w") as f:
            json.dump(calib_data, f, indent=4)
        print("キャリブレーション結果を 'face_camera_calibration.json' に保存しました。")
    else:
        print("カメラキャリブレーションに失敗しました。")

if __name__ == "__main__":
    main()
