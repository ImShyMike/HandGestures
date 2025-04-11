"""An example that makes it possible to use your hands as a mouse."""

import colorsys
from typing import Literal

import cv2
from pynput.mouse import Button, Controller

from hand_gestures.hand_gestures import Condition, GestureBuilder, GestureHandler

mouse = Controller()

screen_x, screen_y = 1920, 1080  # pyautogui.size()
window_width, window_height = (720, 480)


class Gestures:
    """List of gestures."""

    NONE = "none"
    FIST = "fist"
    GRAB = "grab"
    OK = "ok"
    SCROLL = "scroll"
    SCROLL_UP = "scroll up"
    SCROLL_DOWN = "scroll down"
    TWO_UP = "two up"


handler = GestureHandler(
    0,
    screen_size=(screen_x, screen_y),
    window_size=(window_width, window_height),
    _blank_frame=True,
)


def invert_hand(hand) -> Literal["Right", "Left"]:
    """Invert hand."""

    return "Right" if hand == "Left" else "Left"


hand_states = {
    "Left": {"new_y_bkp": 0, "new_x_bkp": 0, "moved": False, "grab_active": False},
    "Right": {"new_y_bkp": 0, "new_x_bkp": 0, "moved": False, "grab_active": False},
}


def move_mouse(median_point, hand) -> None:
    """Function to move the mouse when GRAB gesture is detected."""
    try:
        rel_x = max(0.0, min(1.0, median_point[0]))
        rel_y = max(0.0, min(1.0, median_point[1]))

        if not hand_states[hand]["grab_active"]:
            hand_states[hand]["new_x_bkp"] = rel_x
            hand_states[hand]["new_y_bkp"] = rel_y
            hand_states[hand]["grab_active"] = True
            return

        dx = rel_x - hand_states[hand]["new_x_bkp"]
        dy = rel_y - hand_states[hand]["new_y_bkp"]

        movement_threshold = 0.003
        hand_states[hand]["moved"] = (
            abs(dx) > movement_threshold or abs(dy) > movement_threshold
        )

        if hand_states[hand]["moved"]:
            sensitivity = 2.0
            screen_dx = int(dx * screen_x * sensitivity)
            screen_dy = int(dy * screen_y * sensitivity)

            print(
                f"{hand} hand - dx: {screen_dx}, dy: {screen_dy}, moved: {hand_states[hand]['moved']}"
            )

            mouse.move(screen_dx, screen_dy)

            hand_states[hand]["new_x_bkp"] = rel_x
            hand_states[hand]["new_y_bkp"] = rel_y

    except Exception as e:
        print(f"Error in move_mouse: {e}")


def move_mouse_callback(callback_dict) -> None:
    hand = callback_dict[0]["hand"]
    move_mouse(callback_dict[0]["condition_median_point"], hand)


def scroll_callback(callback_dict) -> None:
    hand = callback_dict[0]["hand"]
    if hand == "Left":
        mouse.scroll(0, 1)
    else:
        mouse.scroll(0, -1)


button_states = {"Left": False, "Right": False, "Middle": False}


def main_callback_func(hand, detected_gestures, hand_scale, hand_positions) -> None:
    if Gestures.GRAB not in detected_gestures:
        hand_states[hand]["new_x_bkp"] = 0
        hand_states[hand]["new_y_bkp"] = 0
        hand_states[hand]["moved"] = False
        hand_states[hand]["grab_active"] = False

    if not detected_gestures:
        handler.set_gesture(hand, Gestures.NONE)

    else:
        if Gestures.FIST in detected_gestures:
            if not button_states[hand]:
                button_states[hand] = True
                if hand == "Left":
                    mouse.press(Button.left)
                else:
                    mouse.press(Button.right)
            handler.set_gesture(hand, Gestures.FIST)
        else:
            if button_states[hand]:
                button_states[hand] = False
                if hand == "Left":
                    mouse.release(Button.left)
                else:
                    mouse.release(Button.right)
            handler.set_gesture(hand, Gestures.OK)

            if Gestures.GRAB in detected_gestures:
                handler.set_gesture(hand, Gestures.GRAB)
            elif Gestures.SCROLL in detected_gestures:
                if hand == "Left":
                    handler.set_gesture(hand, Gestures.SCROLL_UP)
                else:
                    handler.set_gesture(hand, Gestures.SCROLL_DOWN)
            elif Gestures.TWO_UP in detected_gestures:
                handler.set_gesture(hand, Gestures.TWO_UP)


color_value = 0

def frame_modify_callback(frame):
    global color_value
    new_color = colorsys.hsv_to_rgb((color_value % 360) / 360.0, 1, 1)
    new_color = (new_color[0] * 255, new_color[1] * 255, new_color[2] * 255)
    color_value += 1.5
    cv2.putText(
        frame,
        f"x: {str(mouse.position[0]).ljust(4)}, y: {mouse.position[1]}",
        (window_width - 360, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        new_color,
        1,
        cv2.LINE_AA,
    )
    return frame


ok = GestureBuilder(Gestures.OK, [Condition(0, [8, 12, 16, 20], 0.15)])

fist = GestureBuilder(Gestures.FIST, [Condition(4, 6, 0.05)])

grab = GestureBuilder(
    Gestures.GRAB, [Condition(4, 8, 0.06)], callback=move_mouse_callback
)

scroll = GestureBuilder(
    Gestures.SCROLL, [Condition(4, 12, 0.05)], callback=scroll_callback
)

two_up = GestureBuilder(
    Gestures.TWO_UP,
    [
        Condition(4, [15, 19], 0.1),
        Condition(8, 12, 0.05, invert_check=True),
    ],
)

handler.init(
    [fist, ok, grab, scroll, two_up], main_callback_func, frame_modify_callback
)
