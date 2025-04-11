"""Hand gesture recognition using MediaPipe Hands."""

import time
from typing import Any, Callable, List, Literal, Optional, Union

import cv2
import mediapipe as mp
import numpy as np

# Landmark numbers:
# https://mediapipe.dev/images/mobile/hand_landmarks.png


class Condition:
    """Condition class for hand gesture recognition. (used in GestureBuilder)"""

    def __init__(
        self,
        landmark1: int,
        landmark2: Union[int, List[int]],
        distance: float,
        invert_check: bool = False,
        scale: float = 10,
        hand: Literal["Left", "Right", "none"] = "none",
    ) -> None:
        """Condition class for hand gesture recognition. (used in GestureBuilder)

        Args:
            landmark1 (int): ID of the first landmark.
            landmark2 (Union[int, List[int]]): ID (or list of IDs) of the second landmark.
            distance (float): Distance threshold for the condition. (0.0 - 1.0)
            invert_check (bool, optional): Whether or not  to invert the check. Defaults to False.
            scale (float, optional): Distance scaling based on hand size. Defaults to 10.
            hand (Literal["Left", "Right", "none"], optional): Which hand to use. Defaults to "none".

        Raises:
            ValueError: If landmark1 or landmark2 is not between 0 and 20.
            ValueError: If landmark2 is a list and contains values not between 0 and 20.
        """
        if 0 > landmark1 > 20:
            raise ValueError(
                f"Invalid landmarks, landmark1 must be between 0 and 20 (given: {landmark1})"
            )

        if isinstance(landmark2, list):
            if not all(0 <= x <= 20 for x in landmark2):
                raise ValueError(
                    f"Invalid landmarks, landmark2 must be between 0 and 20 (given: {landmark2})"
                )

        elif isinstance(landmark2, int):
            if 0 > landmark2 > 20:
                raise ValueError(
                    f"Invalid landmarks, landmark2 must be between 0 and 20 (given: {landmark2})"
                )

        self.landmark1 = landmark1
        self.landmark2 = landmark2
        self.distance = distance
        self.scale = scale
        self.hand = hand
        self.reversed = invert_check


class GestureBuilder:
    """Builder class for hand gesture recognition. (used in GestureHandler)"""

    def __init__(
        self,
        gesture: str,
        conditions: List[Condition],
        callback: Optional[Callable] = None,
        set_gesture: bool = False,
    ) -> None:
        """Builder class for hand gesture recognition. (used in GestureHandler)

        Args:
            gesture (str): Gesture name.
            conditions (List[Condition]): List of conditions for the gesture.
            callback (Optional[Callable], optional): Callback for the gesture. Defaults to None.
            set_gesture (bool, optional): Whether or not the gesture can set itself as the gesture the hand is doing. If False, you will have to et it based on 'callback(detected_gestures)'. Useful if many gestures are detected at once. Defaults to False.
        """
        self.gesture = gesture
        self.conditions = conditions
        self.callback = callback
        self.set_gesture = set_gesture


class GestureHandler:
    """Main class for hand gesture recognition."""

    class Hand:
        """Simple hand class."""

        def __init__(self, gesture_hand: Literal["Left", "Right"]) -> None:
            """Simple hand class.

            Args:
                gesture_hand (Literal["Left", "Right"]): Which hand it is.
            """
            self.hand = gesture_hand
            self.gesture = "none"

    def __init__(
        self,
        capture=0,
        mp_hands=mp.solutions.hands.Hands(),
        window=True,
        screen_size: tuple[int, int] = (1920, 1080),
        window_size=(720, 480),
        resize_capture=(0, 0),  # type: ignore
        window_name="Hand Recognition",
        exit_key="q",
        render_gesture_name=True,
        render_hand_bounding_box=True,
        render_fps=True,
        render_hand_landmarks=True,
        landmarks_draw_settings=mp.solutions.drawing_styles.get_default_hand_landmarks_style(),  # type: ignore
        connections_draw_settings=mp.solutions.drawing_styles.get_default_hand_connections_style(),  # type: ignore
        _render_scale_line=False,
        _blank_frame=False,
    ) -> None:
        """Main class for hand gesture recognition.

        Args:
            capture (int, optional): What OpenCV capture to use. Defaults to 0 (camera).
            mp_hands (_type_, optional): Custom mediapipe hands object. Defaults to mp.solutions.hands.Hands().
            window (bool, optional): Whether or not to display the output. Defaults to True.
            window_size (tuple, optional): Screen size for the output display. Defaults to (720, 480).
            resize_capture (tuple, optional): Size to resize the capture before processing. Defaults to (0, 0) (disabled).
            window_name (str, optional): Name of the window to display. Defaults to "Hand Recognition".
            exit_key (str, optional): Which key to use to exit the display output. Defaults to "q".
            render_gesture_name (bool, optional): Whether or not to render the names of the gestures. Defaults to True.
            render_hand_bounding_box (bool, optional): Whether or not to render a bounding box around the hands. Defaults to True.
            render_fps (bool, optional): Whether or not to render current FPS. Defaults to True.
            render_hand_landmarks (bool, optional): Whether or not to render hand landmarks. Defaults to True.
            landmarks_draw_settings (_type_, optional): Custom mediapipe landmark draw style. Defaults to mp.solutions.drawing_styles.get_default_hand_landmarks_style().
            connections_draw_settings (_type_, optional): Custom mediapipe connection draw style. Defaults to mp.solutions.drawing_styles.get_default_hand_connections_style().
            _render_scale_line (bool, optional): (DEBUG) Whether or not to render a line between the hand scale landmarks. Defaults to False.
            _blank_frame (bool, optional): (DEBUG) Whether or not to render a blank frame. Defaults to False.
        """
        self.capture = capture
        self.resize_capture = resize_capture
        self.initialized = False
        self._blank_frame = _blank_frame

        self.mp_hands = mp_hands
        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore
        self.hand_solutions = mp.solutions.hands  # type: ignore

        self.screen = window
        self.screen_size = window_size
        self.screen_name = window_name

        self.true_screen_size = screen_size

        self.render_hand_landmarks = render_hand_landmarks
        self.render_gesture_name = render_gesture_name
        self.render_hand_bounding_box = render_hand_bounding_box
        self.render_fps = render_fps
        self._render_scale_line = _render_scale_line

        self.landmarks_draw_settings = landmarks_draw_settings
        self.connections_draw_settings = connections_draw_settings

        self.exit_key = exit_key

        # Initializing stuff

        self.main_callback = None
        self.frame_modify_callback = None

        self.screen_x, self.screen_y = None, None

        self.left_hand = None
        self.right_hand = None

        self.cap = None
        self.frame = None
        self.frame_rgb = None

        self.hand_scale = None

    def distance_3d(self, coord1: dict, coord2: dict) -> Any:
        """3D distance between two coordinates."""
        return (
            (coord1["x"] - coord2["x"]) ** 2
            + (coord1["y"] - coord2["y"]) ** 2
            + (coord1["z"] - coord2["z"]) ** 2
        ) ** 0.5

    def distance_2d(self, coord1: dict, coord2: dict) -> Any:
        """2D distance between two coordinates."""
        return (
            (coord1["x"] - coord2["x"]) ** 2 + (coord1["y"] - coord2["y"]) ** 2
        ) ** 0.5

    def median_point(self, coord1: dict, coord2: dict) -> tuple[Any, Any]:
        """Median point between two coordinates."""
        x_median = (coord1["x"] + coord2["x"]) / 2
        y_median = (coord1["y"] + coord2["y"]) / 2
        return x_median, y_median

    def scale_distance(self, scale: float, distance: float, extra: float = 10) -> float:
        """Scale the distance based on the hand size."""
        if not extra:
            return distance
        scaled_distance = scale * distance * extra
        return scaled_distance

    def set_gesture(
        self, hand: Literal["Left", "Right"], gesture: Union[str, None]
    ) -> bool:
        """Set the main gesture for the hand."""
        if not gesture:
            gesture = "none"
        if self.initialized:
            if hand == "Left":
                self.left_hand.gesture = gesture
                return True

            self.right_hand.gesture = gesture
            return True
        return False

    def get_gesture(
        self, hand: Literal["Left", "Right"]
    ) -> Literal["Left", "Right", "none", False] | str:
        """Get the main gesture for the hand."""
        if self.initialized:
            if hand == "Left":
                return self.left_hand.gesture or "none"

            return self.right_hand.gesture or "none"
        return False

    def init(
        self,
        gestures: list[GestureBuilder],
        callback: Callable,
        frame_modify_callback: Optional[Callable] = None,
    ) -> None:
        """Initialize the gesture handler.
        This function will start the main loop and start processing frames.

        Args:
            screen_size (tuple[int, int], optional): Size of the screen to use. Defaults to (1920, 1080).
            gestures (list[GestureBuilder]): List of gestures to recognize.
            callback (Callable): Main callback function for the gestures.
            frame_modify_callback (Optional[Callable], optional): Callback to modify the frame before displaying. Defaults to None.

        Raises:
            ValueError: If mp_hands is not a mediapipe.solutions.hands.Hands object.
        """

        if not isinstance(self.mp_hands, mp.solutions.hands.Hands):
            raise ValueError(
                "MediaPipe Hands is not a mediapipe.solutions.hands.Hands object"
            )

        self.initialized = True
        self.main_callback = callback
        self.frame_modify_callback = frame_modify_callback

        self.screen_x, self.screen_y = self.true_screen_size  # pyautogui.size()

        self.left_hand = GestureHandler.Hand("Left")
        self.right_hand = GestureHandler.Hand("Right")

        self.cap = cv2.VideoCapture(self.capture)
        self.cap.set(3, self.screen_size[0])
        self.cap.set(4, self.screen_size[1])

        self.mp_drawing = mp.solutions.drawing_utils  # type: ignore

        fps_previous_time = time.time()

        while self.cap.isOpened():
            ret, self.frame = self.cap.read()

            if not ret:
                continue

            if self.resize_capture != (0, 0):
                self.frame = cv2.resize(self.frame, self.resize_capture)

            self.frame = cv2.flip(self.frame, 1)

            frame_height, frame_width, _ = self.frame.shape

            self.frame_rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)

            results = self.mp_hands.process(self.frame_rgb)

            hands_type = []
            if results.multi_handedness:
                for hand in results.multi_handedness:
                    hand_type = hand.classification[0].label
                    hands_type.append(hand_type)

            hand_positions = {"Right": {}, "Left": {}}
            calculate_gestures = []

            if self._blank_frame:
                self.frame = np.zeros((frame_height, frame_width, 3), np.uint8)

            if results.multi_hand_landmarks:
                hand_num = -1
                for hand_landmarks in results.multi_hand_landmarks:
                    hand_num += 1

                    landmarks_are_valid = all(
                        0.01 < landmark.x < 0.99 and 0.01 < landmark.y < 0.99
                        for landmark in hand_landmarks.landmark
                    )
                    if landmarks_are_valid:
                        calculate_gestures.append(hands_type[hand_num])

                        landmark_number = 0
                        x_list, y_list = [], []
                        for landmark in hand_landmarks.landmark:
                            x_list.append(int(landmark.x * frame_width))
                            y_list.append(int(landmark.y * frame_height))
                            hand_positions[hands_type[hand_num]][landmark_number] = {
                                "x": landmark.x,
                                "y": landmark.y,
                                "z": landmark.z,
                            }
                            landmark_number += 1

                        hand_size_1 = hand_positions[hands_type[hand_num]][17]
                        hand_size_2 = hand_positions[hands_type[hand_num]][5]
                        hand_positions[hands_type[hand_num]]["scale"] = (
                            self.distance_2d(hand_size_1, hand_size_2)
                        )

                        xmin, xmax = min(x_list), max(x_list)
                        ymin, ymax = min(y_list), max(y_list)
                        boxW, boxH = xmax - xmin, ymax - ymin
                        bbox = xmin, ymin, boxW, boxH

                        if self.screen:
                            if self.render_hand_bounding_box:
                                cv2.rectangle(
                                    self.frame,
                                    (bbox[0] - 20, bbox[1] - 20),
                                    (bbox[0] + bbox[2] + 20, bbox[1] + bbox[3] + 20),
                                    (200, 0, 100),
                                    2,
                                )

                            if self.render_gesture_name:
                                cv2.putText(
                                    self.frame,
                                    self.get_gesture(hands_type[hand_num]) or "none",
                                    (bbox[0] - 20, bbox[1] - 30),
                                    cv2.FONT_HERSHEY_PLAIN,
                                    2,
                                    (200, 0, 100),
                                    2,
                                )

                            if self._render_scale_line:
                                cv2.line(
                                    self.frame,
                                    (
                                        int(hand_size_1["x"] * frame_width),
                                        int(hand_size_1["y"] * frame_height),
                                    ),
                                    (
                                        int(hand_size_2["x"] * frame_width),
                                        int(hand_size_2["y"] * frame_height),
                                    ),
                                    (100, 100, 100),
                                    2,
                                )

                            if self.render_hand_landmarks:
                                self.mp_drawing.draw_landmarks(
                                    image=self.frame,
                                    landmark_list=hand_landmarks,
                                    connections=self.hand_solutions.HAND_CONNECTIONS,
                                    landmark_drawing_spec=self.landmarks_draw_settings,
                                    connection_drawing_spec=self.connections_draw_settings,
                                )

            for hand in ("Left", "Right"):
                if hand not in calculate_gestures:
                    self.set_gesture(hand, None)

            for hand in calculate_gestures:
                detected_gestures = []
                for gesture in gestures:
                    valid = []
                    callback_items = []
                    for condition in gesture.conditions:
                        if (
                            condition.hand
                            and condition.hand != hand
                            and condition.hand != "none"
                        ):
                            continue
                        if isinstance(condition.landmark1, list) and isinstance(
                            condition.landmark2, list
                        ):
                            raise ValueError("Only one of the landmarks can be a list")
                        if not (
                            isinstance(condition.landmark1, int)
                            or isinstance(condition.landmark2, int)
                        ):
                            raise ValueError(
                                "Atleast one landmark of type int is required"
                            )
                        self.hand_scale = hand_positions[hand]["scale"]
                        if isinstance(condition.landmark1, int) and isinstance(
                            condition.landmark2, int
                        ):
                            distance = self.distance_3d(
                                hand_positions[hand][condition.landmark1],
                                hand_positions[hand][condition.landmark2],
                            )
                            trigger_distance = (
                                self.scale_distance(
                                    condition.distance,
                                    self.hand_scale,
                                    extra=condition.scale,
                                )
                                if condition.scale
                                else condition.distance
                            )
                            condition_median_point = self.median_point(
                                hand_positions[hand][condition.landmark1],
                                hand_positions[hand][condition.landmark2],
                            )

                            if distance < trigger_distance:
                                if condition.reversed:
                                    valid.append(False)
                                else:
                                    callback_items.append(
                                        {
                                            "hand": hand,
                                            "distance": distance,
                                            "condition_median_point": condition_median_point,
                                            "scale": self.hand_scale,
                                        }
                                    )
                                    valid.append(True)
                                continue

                            if condition.reversed:
                                callback_items.append(
                                    {
                                        "hand": hand,
                                        "distance": distance,
                                        "condition_median_point": condition_median_point,
                                        "scale": self.hand_scale,
                                    }
                                )
                                valid.append(True)
                            else:
                                valid.append(False)
                            continue

                        if isinstance(condition.landmark2, list):
                            trigger_distance = (
                                self.scale_distance(
                                    condition.distance,
                                    self.hand_scale,
                                    extra=condition.scale,
                                )
                                if condition.scale
                                else condition.distance
                            )
                            triggered = all(
                                self.distance_3d(
                                    hand_positions[hand][condition.landmark1],
                                    hand_positions[hand][_landmark],
                                )
                                < trigger_distance
                                for _landmark in condition.landmark2
                            )
                            if triggered:
                                if condition.reversed:
                                    valid.append(False)
                                else:
                                    callback_items.append(
                                        {"hand": hand, "scale": self.hand_scale}
                                    )
                                    valid.append(True)
                                continue

                            if condition.reversed:
                                callback_items.append(
                                    {"hand": hand, "scale": self.hand_scale}
                                )
                                valid.append(True)
                            else:
                                valid.append(False)
                            continue
                        raise ValueError("Invalid condition")
                    if valid and all(valid):
                        detected_gestures.append(gesture.gesture)
                        if gesture.set_gesture:
                            if len(detected_gestures) > 0:
                                self.set_gesture(hand, detected_gestures[0])
                        if gesture.callback:
                            if len(callback_items) > 0:
                                gesture.callback(callback_items)
                if len(detected_gestures) < 1:
                    if gesture.set_gesture:  # type: ignore
                        self.set_gesture(hand, None)
                self.main_callback(
                    hand, detected_gestures, self.hand_scale, hand_positions[hand]
                )

            if self.render_fps and self.screen:
                fps_current_time = time.time()
                fps = 1 / (fps_current_time - fps_previous_time)
                fps_previous_time = fps_current_time
                cv2.putText(
                    self.frame,
                    f"FPS: {int(fps)}",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 230, 0),
                    2,
                )

            if self.screen:
                if self.frame_modify_callback:
                    self.frame = self.frame_modify_callback(self.frame)
                cv2.imshow(self.screen_name, self.frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord(self.exit_key):
                break

        self.cap.release()
        cv2.destroyAllWindows()
