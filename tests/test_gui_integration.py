import sys
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import pytest

FAKE_SESSION = {"frames": [], "summary": "test"}


def _make_interpreter_mock():
    """Return a mock Interpreter class that creates properly configured instances."""
    instances = []

    def _init(*args, **kwargs):
        inst = MagicMock()
        instances.append(inst)

        if len(instances) == 1:
            # MoveNet
            inst.get_input_details.return_value = [{"index": 0, "shape": [1, 256, 256, 3]}]
            inst.get_output_details.return_value = [{"index": 0, "shape": [1, 1, 17, 3]}]

            def _get_tensor(idx):
                kps = np.zeros((1, 1, 17, 3), dtype=np.float32)
                kps[0, 0, :, 2] = 0.9
                return kps

            inst.get_tensor.side_effect = _get_tensor
        else:
            # Classifier
            inst.get_input_details.return_value = [{"index": 0, "shape": [1, 30, 24]}]
            inst.get_output_details.return_value = [
                {"index": 1, "shape": [1, 10]},
                {"index": 2, "shape": [1, 2]},
            ]

            def _get_tensor(idx):
                if idx == 1:
                    out = np.zeros((1, 10), dtype=np.float32)
                    out[0, 0] = 0.9
                    return out
                elif idx == 2:
                    out = np.zeros((1, 2), dtype=np.float32)
                    out[0, 1] = 0.9
                    return out
                return np.zeros((1, 10), dtype=np.float32)

            inst.get_tensor.side_effect = _get_tensor

        return inst

    mock_cls = MagicMock(side_effect=_init)
    return mock_cls


@pytest.fixture
def gui_mod():
    """Import gui.py with all heavy dependencies mocked out."""
    if "gui" in sys.modules:
        del sys.modules["gui"]

    # Mocks for externally-imported classes/functions
    mock_tts_class = MagicMock()
    mock_tts_instance = MagicMock()
    mock_tts_class.return_value = mock_tts_instance

    mock_logger_class = MagicMock()
    mock_logger_instance = MagicMock()
    mock_logger_instance.end_session.return_value = FAKE_SESSION
    mock_logger_class.return_value = mock_logger_instance

    mock_start_server = MagicMock()
    mock_qrcode = MagicMock()
    mock_qr_img = MagicMock()
    mock_qrcode.make.return_value = mock_qr_img

    fake_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    mock_cap = MagicMock()
    mock_cap.read.return_value = (True, fake_frame)

    mock_interpreter = _make_interpreter_mock()

    # Build mock cv2 module manually so we never import the real (broken) cv2
    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.resize.side_effect = lambda img, size: np.zeros((size[1], size[0], 3), dtype=np.uint8)
    mock_cv2.cvtColor.side_effect = lambda img, code: img
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_BUFFERSIZE = 1
    mock_cv2.COLOR_BGR2RGB = 4

    # Build mock tkinter module
    mock_tk = MagicMock()
    mock_tk.Tk.return_value = MagicMock()
    mock_tk.Frame.return_value = MagicMock()
    mock_tk.Label.return_value = MagicMock()
    mock_tk.Button.return_value = MagicMock()
    mock_tk.Canvas.return_value = MagicMock()

    # Build mock PIL modules
    mock_pil_image = MagicMock()
    mock_pil_image.fromarray.return_value = MagicMock()
    mock_pil_imagetk = MagicMock()
    mock_pil_imagetk.PhotoImage.return_value = MagicMock()
    mock_pil = MagicMock()
    mock_pil.Image = mock_pil_image
    mock_pil.ImageTk = mock_pil_imagetk

    # Build mock tflite / litert interpreter modules
    mock_tflite_interp = MagicMock()
    mock_tflite_interp.Interpreter = mock_interpreter
    mock_litert_interp = MagicMock()
    mock_litert_interp.Interpreter = mock_interpreter

    extra_modules = {
        "tts_engine": MagicMock(),
        "session_logger": MagicMock(),
        "session_chat": MagicMock(),
        "session_chat.app": MagicMock(),
        "qrcode": mock_qrcode,
        "tkinter": mock_tk,
        "PIL": mock_pil,
        "cv2": mock_cv2,
        "tflite_runtime": MagicMock(),
        "tflite_runtime.interpreter": mock_tflite_interp,
        "ai_edge_litert": MagicMock(),
        "ai_edge_litert.interpreter": mock_litert_interp,
    }
    extra_modules["tts_engine"].TTSEngine = mock_tts_class
    extra_modules["session_logger"].SessionLogger = mock_logger_class
    extra_modules["session_chat.app"].start_server = mock_start_server

    with patch.dict(sys.modules, extra_modules):
        import gui

        # Expose key mocks for test assertions
        gui._test_mocks = {
            "tts_class": mock_tts_class,
            "tts_instance": mock_tts_instance,
            "logger_class": mock_logger_class,
            "logger_instance": mock_logger_instance,
            "start_server": mock_start_server,
            "qrcode": mock_qrcode,
            "qr_img": mock_qr_img,
        }
        yield gui

    if "gui" in sys.modules:
        del sys.modules["gui"]


def test_tts_initialized_on_startup(gui_mod):
    """TTSEngine should be instantiated and announce_session_start called once on import."""
    gui_mod._test_mocks["tts_instance"].announce_session_start.assert_called_once()


def test_logger_initialized_on_startup(gui_mod):
    """SessionLogger should be instantiated once on import."""
    assert gui_mod._test_mocks["logger_class"].call_count == 1


def _run_one_inference_iteration(gui_mod):
    """Pre-fill rolling buffers and trigger one update() so smoothing fires."""
    dummy_frame = np.zeros((12, 2), dtype=np.float32)
    for _ in range(30):
        gui_mod.frame_buffer.append(dummy_frame)

    for _ in range(5):
        gui_mod.prediction_buffer.append((0, 1, 0.9))

    gui_mod.frame_counter = 4

    gui_mod.tts.update.reset_mock()
    gui_mod.logger.log_frame.reset_mock()

    gui_mod.update()


def test_inference_loop_calls_tts_update(gui_mod):
    """tts.update should be called with the expected exercise name, quality, and a timestamp."""
    _run_one_inference_iteration(gui_mod)
    gui_mod.tts.update.assert_called_once_with("Deep Squat", 1, ANY)


def test_inference_loop_calls_logger_log_frame(gui_mod):
    """logger.log_frame should be called with the expected indices and confidence."""
    _run_one_inference_iteration(gui_mod)
    gui_mod.logger.log_frame.assert_called_once()
    args = gui_mod.logger.log_frame.call_args[0]
    assert args[0] == 0
    assert args[1] == 1
    assert args[2] == pytest.approx(0.9)


def test_end_session_calls_logger_end_session(gui_mod):
    """_end_session should call logger.end_session exactly once."""
    gui_mod._end_session()
    gui_mod.logger.end_session.assert_called_once()


def test_end_session_calls_start_server(gui_mod):
    """_end_session should call start_server with the session data returned by logger."""
    gui_mod._end_session()
    gui_mod.start_server.assert_called_once_with(FAKE_SESSION)


def test_end_session_displays_qr_code(gui_mod):
    """_end_session should generate a QR code for the review URL."""
    gui_mod._end_session()
    gui_mod.qrcode.make.assert_called_once()
    url_passed = gui_mod.qrcode.make.call_args[0][0]
    assert ":5000" in url_passed
