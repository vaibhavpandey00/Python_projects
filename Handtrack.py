# #Import the libraries
# #Installation:openCV and mediapipe
# import cv2
# import mediapipe as mp
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles
# mphands = mp.solutions.hands

# cap = cv2.VideoCapture(0)
# hands = mphands.Hands(
#     model_complexity=0,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )

# while True:
#     data, image = cap.read()
#     #Convert the BGR image to RGB
#     image = cv2.cvtColor(cv2.flip(image,1), cv2.COLOR_BGR2RGB)
#     results = hands.process(image)
#     image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 image,
#                 hand_landmarks,
#                 mphands.HAND_CONNECTIONS,
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )
#     cv2.imshow('Hand Tracking', image)
#     cv2.waitKey(1)



# 2nd code
# import cv2
# import mediapipe as mp
# import numpy as np
# import pyautogui

# # Initialize Mediapipe hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     max_num_hands=1,  # Limit to only one hand
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
# mp_drawing = mp.solutions.drawing_utils

# # Function to control mouse cursor
# def control_mouse(x, y):
#     screen_width, screen_height = pyautogui.size()
#     pyautogui.moveTo(int(x * screen_width), int(y * screen_height))

# # Main loop
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Flip the image horizontally (from right to left)
#     frame = cv2.flip(frame, 1)

#     # Convert BGR image to RGB
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Process image
#     results = hands.process(image_rgb)

#     # Draw landmarks and control mouse
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Check if the detected hand is the right hand
#             if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x < hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x:
#                 # Draw hand landmarks
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Get index finger tip coordinates
#                 index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 index_finger_tip_x = index_finger_tip.x
#                 index_finger_tip_y = index_finger_tip.y

#                 # Control mouse cursor
#                 control_mouse(index_finger_tip_x, index_finger_tip_y)

#     # Display frame
#     cv2.imshow('Hand Tracking', frame)

#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()


# 3rd code
# import cv2
# import mediapipe as mp
# import numpy as np
# import pyautogui
# import math

# # Initialize Mediapipe hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     max_num_hands=1,  # Limit to only one hand
#     min_detection_confidence=0.2,
#     min_tracking_confidence=0.2
# )
# mp_drawing = mp.solutions.drawing_utils

# # Initialize variables for previous cursor position
# prev_x = 0
# prev_y = 0

# # Function to control mouse cursor and register left click  (, pinch_distance)
# def control_mouse(x, y):
#     global prev_x, prev_y
#     screen_width, screen_height = pyautogui.size()
#     # Smooth the cursor movement by averaging with previous position
#     x_smooth = (x + prev_x) / 2
#     y_smooth = (y + prev_y) / 2
#     pyautogui.moveTo(int(x_smooth * screen_width), int(y_smooth * screen_height))
#     prev_x = x
#     prev_y = y
#     # Register left click if pinch distance is below a certain threshold
#     # if pinch_distance < 0.03:  # Adjust this threshold according to your preference
#     #     pyautogui.click()

# # Calculate Euclidean distance between two points
# # def calculate_distance(point1, point2):
# #     return math.sqrt((point1.x - point2.x)**2 + (point1.y - point2.y)**2)

# # Main loop
# cap = cv2.VideoCapture(0)
# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Flip the image horizontally (from right to left)
#     frame = cv2.flip(frame, 1)

#     # Convert BGR image to RGB
#     image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
#     # Process image
#     results = hands.process(image_rgb)

#     # Draw landmarks and control mouse
#     if results.multi_hand_landmarks:
#         for hand_landmarks in results.multi_hand_landmarks:
#             # Check if the detected hand is the right hand
#             if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x:
#                 # Draw hand landmarks
#                 mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#                 # Get index finger tip and thumb coordinates
#                 index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 # thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

#                 # Calculate distance between index finger tip and thumb
#                 # pinch_distance = calculate_distance(index_finger_tip, thumb_tip)

#                 # Get index finger tip coordinates
#                 index_finger_tip_x = index_finger_tip.x
#                 index_finger_tip_y = index_finger_tip.y

#                 # Control mouse cursor and register left click   (, pinch_distance)
#                 control_mouse(index_finger_tip_x, index_finger_tip_y)

#     # Display frame
#     cv2.imshow('Hand Tracking', frame)

#     # Exit if 'q' is pressed
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()




# 4th code
# import cv2
# import mediapipe as mp
# import numpy as np
# import pyautogui
# import time

# # Initialize Mediapipe hands module
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(
#     max_num_hands=1,  # Limit to only one hand
#     min_detection_confidence=0.3,
#     min_tracking_confidence=0.3
# )
# mp_drawing = mp.solutions.drawing_utils

# # Main loop
# cap = cv2.VideoCapture(0)

# # Initialize play/pause flag
# is_playing = False

# # Initialize cooldown timer
# cooldown_start_time = 0
# cooldown_duration = 1.0  # Adjust the cooldown duration as needed

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     # Flip the image horizontally (from right to left)
#     frame = cv2.flip(frame, 1)

#     # Process image
#     output = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Create a black image with the same size as the frame
#     black_image = np.zeros_like(frame)

#     # Draw landmarks
#     if output.multi_hand_landmarks:
#         for hand_landmarks in output.multi_hand_landmarks:

#             # Check if the detected hand is the right hand
#             if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x:

#                 # Draw hand landmarks
#                 mp_drawing.draw_landmarks(
#                     black_image, 
#                     hand_landmarks, 
#                     mp_hands.HAND_CONNECTIONS
#                     )

#                 # Get index finger tip and thumb coordinates
#                 middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#                 index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

#                 # Calculate distance between index finger tip and thumb
#                 distance = np.linalg.norm([index_finger_tip.x - thumb_tip.x, index_finger_tip.y - thumb_tip.y])
#                 distancebtindexandmiddle = np.linalg.norm([index_finger_tip.x - middle_finger_tip.x, index_finger_tip.y - middle_finger_tip.y])

#                 # Print distance
#                 print("Distance between index finger tip and thumb tip: {:.2f}".format(distance))

#                 # Play/pause media if distance is smaller than or equal to 0.03
#                 if distancebtindexandmiddle > 0.15:
#                     if distance <= 0.03 and (time.time() - cooldown_start_time) > cooldown_duration:
#                         if not is_playing:
#                             pyautogui.press('playpause')
#                             is_playing = True
#                         else:
#                             pyautogui.press('playpause')
#                             is_playing = False
#                         cooldown_start_time = time.time()  # Update cooldown start time

#             # Combine black image and frame
#             else:
#                 frame = cv2.bitwise_and(frame, black_image)
#     else:
#         frame = cv2.bitwise_and(frame, black_image)

#     # Display frame
#     cv2.imshow('Hand Tracking', frame)

#     # Exit if 'CTRL' + 'Shift' + 'Q' are pressed
#     if cv2.waitKey(1) & 0xFF == ord('Q') and cv2.waitKey(1) & 0xFF == ord('q') and cv2.waitKey(1) & 0xFF == ord('\x11'):
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()



# 5th code
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time

# Initialize Mediapipe hands module
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,  # Limit to only one hand
    min_detection_confidence=0.3,
    min_tracking_confidence=0.3
)
mp_drawing = mp.solutions.drawing_utils

# Video capture
cap = cv2.VideoCapture(0)

# Initialize play/pause flag
is_playing = False

# Initialize cooldown timer
cooldown_start_time = 0
cooldown_duration = 1.0  # In seconds

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Flip the image horizontally (from right to left)
    frame = cv2.flip(frame, 1)

    # Process image
    output = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    # Create a black image with the same size as the frame
    black_image = np.zeros_like(frame)

    # Draw landmarks on the black image if right hand is detected
    if output.multi_hand_landmarks:
        for hand_landmarks in output.multi_hand_landmarks:

            # Check if the detected hand is the right hand
            if hand_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x:

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    black_image, 
                    hand_landmarks, 
                    mp_hands.HAND_CONNECTIONS
                    )

                # Get index finger tip and thumb coordinates
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                # Calculate distance between index finger tip and thumb
                distance = np.linalg.norm([index_finger_tip.x - thumb_tip.x, index_finger_tip.y - thumb_tip.y])
                distancebtindexandmiddle = np.linalg.norm([index_finger_tip.x - middle_finger_tip.x, index_finger_tip.y - middle_finger_tip.y])

                # Print distance
                print("Distance between index finger tip and thumb tip: {:.2f}".format(distance))

                # Play/pause media if distance is smaller than or equal to 0.03
                if distancebtindexandmiddle > 0.15:
                    if distance <= 0.03 and (time.time() - cooldown_start_time) > cooldown_duration:
                        if not is_playing:
                            pyautogui.press('playpause')
                            print('Clicked')
                            is_playing = True
                        else:
                            pyautogui.press('playpause')
                            print('Clicked')
                            is_playing = False
                        cooldown_start_time = time.time()  # Update cooldown start time

    # Display the black image if no hand is detected
    frame = black_image

    # Display frame
    cv2.imshow('Hand Tracking', frame)

    #  Exit if "ESC" key is pressed
    if cv2.waitKey(1) == 27:
        cap.release()
        cv2.destroyAllWindows()
        break

# Release resources
cap.release()
cv2.destroyAllWindows()



# 6th code Combined code for Handtrack.py and FacwDetector.py

# import cv2
# import mediapipe as mp
# import numpy as np
# import pyautogui
# import time

# # Video capture
# cap = cv2.VideoCapture(0)

# mp_hands = mp.solutions.hands
# facemesh = mp.solutions.face_mesh
# hands = mp_hands.Hands(
#     max_num_hands=1,  # Limit to only one hand
#     min_detection_confidence=0.3,
#     min_tracking_confidence=0.3
# )
# face = facemesh.FaceMesh(
#     static_image_mode=False,
#     max_num_faces=1,
#     refine_landmarks=True,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5
# )
# mp_drawing = mp.solutions.drawing_utils

# # Initialize play/pause flag
# is_playing = False

# # Initialize cooldown timer
# cooldown_start_time = 0
# cooldown_duration = 1.0  # In seconds

# while True:
#     ret, frame = cap.read()  # Capture frame-by-frame
#     if not ret:
#         continue

#     # Flip the image horizontally (from right to left)
#     frame = cv2.flip(frame, 1)

#     # Process hand image
#     hand_output = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Process face image
#     face_output = face.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#     # Create a black image with the same size as the frame
#     black_image = np.zeros_like(frame)

#     # Draw landmarks on the black image if face is detected
#     if face_output.multi_face_landmarks:
#         for face_landmarks in face_output.multi_face_landmarks:

#             # Draw face landmarks
#             mp_drawing.draw_landmarks(
#                 black_image,
#                 face_landmarks,
#                 facemesh.FACEMESH_CONTOURS,
#                 mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
#             )

#             if face_landmarks.landmark[mp_hands.HandLandmark.WRIST].x > face_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x:

#                 # Draw hand landmarks
#                 mp_drawing.draw_landmarks(
#                     black_image, 
#                     face_landmarks, 
#                     mp_hands.HAND_CONNECTIONS
#                     )

#                 # Get index finger tip and thumb coordinates
#                 middle_finger_tip = face_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
#                 index_finger_tip = face_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
#                 thumb_tip = face_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

#                 # Calculate distance between index finger tip and thumb
#                 distance = np.linalg.norm([index_finger_tip.x - thumb_tip.x, index_finger_tip.y - thumb_tip.y])
#                 distancebtindexandmiddle = np.linalg.norm([index_finger_tip.x - middle_finger_tip.x, index_finger_tip.y - middle_finger_tip.y])

#                 # Print distance
#                 print("Distance between index finger tip and thumb tip: {:.2f}".format(distance))

#                 # Play/pause media if distance is smaller than or equal to 0.03
#                 if distancebtindexandmiddle > 0.15:
#                     if distance <= 0.03 and (time.time() - cooldown_start_time) > cooldown_duration:
#                         if not is_playing:
#                             pyautogui.press('playpause')
#                             print('Clicked')
#                             is_playing = True
#                         else:
#                             pyautogui.press('playpause')
#                             print('Clicked')
#                             is_playing = False
#                         cooldown_start_time = time.time()  # Update cooldown start time
#     # Display the black image if no hand is detected
#     frame = black_image

#     # Display frame
#     cv2.imshow('Hand Tracking', frame)

#     #  Exit if "ESC" key is pressed
#     if cv2.waitKey(1) == 27:
#         cap.release()
#         cv2.destroyAllWindows()
#         break

# # Release resources
# cap.release()
# cv2.destroyAllWindows()
 