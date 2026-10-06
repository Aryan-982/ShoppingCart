import unittest

from main import add_detections_to_cart


class CartLogicTest(unittest.TestCase):
    def test_image_mode_counts_all_visible_items(self):
        cart = {}
        add_detections_to_cart(cart, ["apple", "banana", "apple"], source="image")
        self.assertEqual(cart, {"apple": 2, "banana": 1})

    def test_video_mode_keeps_tracking_logic(self):
        cart = {}
        add_detections_to_cart(cart, ["apple", "apple"], source="camera")
        self.assertEqual(cart, {})

    def test_duplicate_items_are_counted_once_per_detection_list(self):
        cart = {}
        add_detections_to_cart(cart, ["banana", "banana"], source="image")
        self.assertEqual(cart, {"banana": 2})


if __name__ == "__main__":
    unittest.main()
