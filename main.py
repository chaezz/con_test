import sys
import cv2
import numpy as np
import mediapipe as mp


def frozenset_to_list(frozen_set: frozenset):
    result = []
    data = [list(x) for x in frozen_set]
    for x in data:
        for y in x:
            result.append(y)
    return result


mp_face_mesh = mp.solutions.face_mesh

# left, right iris indices
LEFT_IRIS = [474, 475, 475, 476, 476, 477, 477, 474]
RIGHT_IRIS = [469, 470, 470, 471, 471, 472, 472, 469]
LEFT_EYE = frozenset_to_list(mp_face_mesh.FACEMESH_LEFT_EYE)
RIGHT_EYE = frozenset_to_list(mp_face_mesh.FACEMESH_RIGHT_EYE)
LEFT_EYEBROW = frozenset_to_list(mp_face_mesh.FACEMESH_LEFT_EYEBROW)
RIGHT_EYEBROW = frozenset_to_list(mp_face_mesh.FACEMESH_RIGHT_EYEBROW)
NOSE = frozenset_to_list(mp_face_mesh.FACEMESH_NOSE)
MOUSE = frozenset_to_list(mp_face_mesh.FACEMESH_LIPS)

background = cv2.imread('resource/file_test_resize.jpg')
lens = cv2.imread('resource/lens.png')

cap = cv2.VideoCapture(0)  # WebCam을 읽어들임
with mp_face_mesh.FaceMesh(
        max_num_faces=1,  # 최대로 검출할 얼굴 (int)
        refine_landmarks=True,  # 눈과 입술 주변의 landmark를 정교하게 검출 (bool)
        static_image_mode=True,  # True: 첫 프레임에서 탐지 후 traking으로 진행, False: 모든 프레임 탐지 (bool)
        min_detection_confidence=0.5,  # 최소 탐지 정확도 (float: 0 ~ 1.0)
        min_tracking_confidence=0.5  # 최소 추적 신뢰도. static_image_mode가 True면 무시 (float: 0 ~ 1.0)
) as face_mesh:
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_h, img_w = frame.shape[:2]
            results = face_mesh.process(rgb_frame)
            mask = np.zeros((img_h, img_w), dtype=np.uint8)
            # mask = np.full((img_h, img_w), fill_value=255, dtype=np.uint8)

            if results.multi_face_landmarks:
                mesh_points = np.array([np.multiply([p.x, p.y], [img_w, img_h]).astype(int)
                                        for p in results.multi_face_landmarks[0].landmark])
                # cv2.polylines(frame, [mesh_points[LEFT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)
                # cv2.polylines(frame, [mesh_points[RIGHT_IRIS]], True, (0,255,0), 1, cv2.LINE_AA)

                (l_cx, l_cy), l_radius = cv2.minEnclosingCircle(mesh_points[LEFT_IRIS])
                (r_cx, r_cy), r_radius = cv2.minEnclosingCircle(mesh_points[RIGHT_IRIS])
                center_left = np.array([l_cx, l_cy], dtype=np.int32)
                center_right = np.array([r_cx, r_cy], dtype=np.int32)

                # cv2.circle(frame, center_left, int(l_radius), (0, 255, 0), -1, cv2.LINE_AA)
                # cv2.circle(frame, center_right, int(r_radius), (0, 255, 0), -1, cv2.LINE_AA)
                # Input Lens image

                cv2.circle(mask, center_left, int(l_radius), (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(mask, center_right, int(r_radius), (255, 255, 255), -1, cv2.LINE_AA)

                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                left_contour, right_contour = contours[0], contours[1]
                lx, ly, lw, lh = cv2.boundingRect(left_contour)
                cv2.rectangle(mask, (lx, ly), (lx + lw, ly + lh), (255, 255, 255), 1)
                rx, ry, rw, rh = cv2.boundingRect(right_contour)
                cv2.rectangle(mask, (rx, ry), (rx + rw, ry + rh), (255, 255, 255), 1)

                left_lens, right_lens = cv2.resize(lens, (lw, lh)), cv2.resize(lens, (rw, rh))
                _, left_thr = cv2.threshold(cv2.cvtColor(left_lens, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)
                _, right_thr = cv2.threshold(cv2.cvtColor(right_lens, cv2.COLOR_BGR2GRAY), 1, 255, cv2.THRESH_BINARY)

                # cv2.polylines(mask, [mesh_points[LEFT_EYEBROW]], True, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.polylines(mask, [mesh_points[RIGHT_EYEBROW]], True, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.polylines(mask, [mesh_points[NOSE]], True, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.polylines(mask, [mesh_points[MOUSE]], True, (0, 255, 0), 1, cv2.LINE_AA)
                # cv2.polylines(mask, [mesh_points[LEFT_EYE]], True, (255, 255, 255), 1, cv2.LINE_AA)
                # cv2.polylines(mask, [mesh_points[RIGHT_EYE]], True, (255, 255, 255), 1, cv2.LINE_AA)

                # new_image = cv2.bitwise_or(frame[rx:rx + rw, ry:ry + rh], lens)
                # left_lens = cv2.cvtColor(left_lens, cv2.COLOR_BGR2BGRA)
                # right_lens = cv2.cvtColor(right_lens, cv2.COLOR_BGR2BGRA)
                # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2BGRA)
                # print(mask.shape, left_lens.shape)
                # frame[ly:ly + lh, lx:lx + lw] = left_lens
                # frame[ry:ry + rh, rx:rx + rw] = right_lens
                cv2.copyTo(left_lens, left_thr, frame[ly:ly + lh, lx:lx + lw])
                cv2.copyTo(right_lens, right_thr, frame[ry:ry + rh, rx:rx + rw])
                # print(f'mask[{lx}:{lx + lw}, {ly}:{ly + lh}]')

            # print('lens:', lens.shape)
            # print('mask:', mask.shape)
            # print('background:', background.shape)

            # cv2.copyTo(lens, mask, background)
            # new_image = cv2.bitwise_or(background, mask)

            cv2.imshow('mediapipe', frame)
            cv2.imshow('mask', mask)
            cv2.imshow('thr', left_thr)

            key = cv2.waitKey(1)
            if key == ord('q'):
                break

        except Exception as e:
            print(e)
cap.release()
cv2.destroyAllWindows()
