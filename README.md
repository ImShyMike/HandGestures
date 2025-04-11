# HandGestures

Python library that makes it easy to create and detect custom gestures using mediapipe.

## Example Image

![floating mouse demo](assets/floating_mouse.png)

(the image is using the 'floating_mouse.py' example with the debug option '_blank_frame=True')

## Example Code

```python
from hand_gestures import Condition, GestureBuilder, GestureHandler


class Gestures:
    NONE = "none"
    FIST = "fist"
    GRAB = "grab"
    OK = "ok"
    PINCH = "pinch"
    TWO_UP = "two up"


handler = GestureHandler(0)


def callback_func(callback_dict):
    print(
        f"1 - Hand: {callback_dict[0]['hand']}, Distance: {callback_dict[0]['distance']}, Condition Median Point: {callback_dict[0]['condition_median_point']}"
    )


def main_callback_func(hand, detected_gestures, hand_scale, hand_positions):
    if not detected_gestures:
        handler.set_gesture(hand, Gestures.NONE)
        return

    if Gestures.OK in detected_gestures:
        if Gestures.FIST in detected_gestures:
            handler.set_gesture(hand, Gestures.FIST)
        else:
            handler.set_gesture(hand, Gestures.OK)

    elif Gestures.GRAB in detected_gestures:
        handler.set_gesture(hand, Gestures.GRAB)

    elif Gestures.PINCH in detected_gestures:
        handler.set_gesture(hand, Gestures.PINCH)

    elif Gestures.TWO_UP in detected_gestures:
        handler.set_gesture(hand, Gestures.TWO_UP)


ok = GestureBuilder(Gestures.OK, [Condition(0, [8, 12, 16, 20], 0.15)])
fist = GestureBuilder(Gestures.FIST, [Condition(4, 6, 0.05)])
grab = GestureBuilder(
    Gestures.GRAB, [Condition(4, 8, 0.05, hand="Left")], callback=callback_func
)
pinch = GestureBuilder(Gestures.PINCH, [Condition(4, 12, 0.05, hand="Right")])
two_up = GestureBuilder(
    Gestures.TWO_UP,
    [Condition(4, [16, 20], 0.1), Condition(8, 12, 0.05, invert_check=True)],
)

handler.init([fist, ok, grab, pinch, two_up], main_callback_func)
```
