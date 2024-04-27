# import cv2
# import mediapipe as mp
# import numpy as np

# cap = cv2.VideoCapture(0)

# facmesh = mp.solutions.face_mesh
# face = facmesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
# draw = mp.solutions.drawing_utils

# while True:
#     ret, frame = cap.read()

#     # Flip the image horizontally (from right to left)
#     frame = cv2.flip(frame, 1)

#     output = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
#     if output.multi_face_landmarks:
#         for face_landmarks in output.multi_face_landmarks:
#             draw.draw_landmarks(
#                 frame,
#                 face_landmarks,
#                 facmesh.FACEMESH_CONTOURS,
#                 draw.DrawingSpec(color=(235, 64, 52), thickness=1, circle_radius=1),
#                 )

#     cv2.imshow('Window', frame)

#     if cv2.waitKey(1) == 27:
#         cap.release()
#         cv2.destroyAllWindows()
#         break

# 2nd code
import cv2
import mediapipe as mp
import numpy as np

cap = cv2.VideoCapture(0)

facemesh = mp.solutions.face_mesh
face = facemesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
draw = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()

    # Flip the image horizontally (from right to left)
    frame = cv2.flip(frame, 1)

    output = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # Create a black image with the same size as the frame
    black_image = np.zeros_like(frame)
    
    if output.multi_face_landmarks:
        for face_landmarks in output.multi_face_landmarks:

            # Draw the face mesh landmarks on the black image
            draw.draw_landmarks(
                black_image,
                face_landmarks,
                facemesh.FACEMESH_CONTOURS,
                draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
            )

            # Combine the frame with the black image (face mesh landmarks) using a bitwise AND operation
            frame = cv2.bitwise_and(frame, black_image)
    
    else:
        frame = cv2.bitwise_and(frame, black_image)

    cv2.imshow('Window', frame)

    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break

