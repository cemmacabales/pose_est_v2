import sys
from unittest.mock import MagicMock, patch, ANY
import numpy as np
import pytest

FAKE_SESSION = {"frames": [], "summary": "test"}


def _make_interpreter_mock():
    """Return a mock Interpreter class for the classifier."""
    mock_inst = MagicMock()
    mock_inst.get_input_details.return_value = [{"index": 0, "shape": [1, 30, 16]}]
    mock_inst.get_output_details.return_value = [
        {"index": 1, "shape": [1, 9]},
        {"index": 2, "shape": [1, 2]},
    ]

    def _get_tensor(idx):
        if idx == 1:
            out = np.zeros((1, 9), dtype=np.float32)
            out[0, 0] = 0.9
            return out
        elif idx == 2:
            out = np.zeros((1, 2), dtype=np.float32)
            out[0, 1] = 0.9
            return out
        return np.zeros((1, 9), dtype=np.float32)

    mock_inst.get_tensor.side_effect = _get_tensor
    return MagicMock(return_value=mock_inst)


def _make_pose_estimator_mock():
    """Return a mock PoseEstimator class with fake BlazePose behavior."""
    mock_inst = MagicMock()
    mock_inst.process_frame.return_value = MagicMock()
    mock_inst.submit_frame = MagicMock()
    mock_inst.latest_landmarks = MagicMock()  # non-None so joints get extracted
    mock_inst.draw_landmarks = MagicMock()

    mock_cls = MagicMock(return_value=mock_inst)
    mock_cls.count_visible = MagicMock(return_value=33)
    mock_cls.extract_mapped_joints = MagicMock(
        return_value=np.zeros((12, 2), dtype=np.float32)
    )
    return mock_cls, mock_inst


@pytest.fixture
def gui_mod():
    """Import gui.py with all heavy dependencies mocked out."""
    if "gui" in sys.modules:
        del sys.modules["gui"]

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
    mock_cap.get.side_effect = lambda prop: 640 if prop == 3 else (480 if prop == 4 else 0)

    mock_interpreter = _make_interpreter_mock()
    mock_pose_estimator_cls, mock_pose_estimator_inst = _make_pose_estimator_mock()

    mock_cv2 = MagicMock()
    mock_cv2.VideoCapture.return_value = mock_cap
    mock_cv2.cvtColor.side_effect = lambda img, code: img
    mock_cv2.CAP_PROP_FRAME_WIDTH = 3
    mock_cv2.CAP_PROP_FRAME_HEIGHT = 4
    mock_cv2.CAP_PROP_BUFFERSIZE = 1
    mock_cv2.COLOR_BGR2RGB = 4

    mock_tk = MagicMock()
    mock_tk.Tk.return_value = MagicMock()
    mock_tk.Frame.return_value = MagicMock()
    mock_tk.Label.return_value = MagicMock()
    mock_tk.Button.return_value = MagicMock()
    mock_tk.Canvas.return_value = MagicMock()

    mock_pil_image = MagicMock()
    mock_pil_image.fromarray.return_value = MagicMock()
    mock_pil_imagetk = MagicMock()
    mock_pil_imagetk.PhotoImage.return_value = MagicMock()
    mock_pil = MagicMock()
    mock_pil.Image = mock_pil_image
    mock_pil.ImageTk = mock_pil_imagetk

    mock_tflite_interp = MagicMock()
    mock_tflite_interp.Interpreter = mock_interpreter
    mock_litert_interp = MagicMock()
    mock_litert_interp.Interpreter = mock_interpreter

    mock_pose_estimator_module = MagicMock()
    mock_pose_estimator_module.PoseEstimator = mock_pose_estimator_cls

    extra_modules = {
        "tts_engine": MagicMock(),
        "session_logger": MagicMock(),
        "session_chat": MagicMock(),
        "session_chat.app": MagicMock(),
        "qrcode": mock_qrcode,
        "tkinter": mock_tk,
        "tkinter.messagebox": MagicMock(),
        "tkinter.filedialog": MagicMock(),
        "tkinter.ttk": MagicMock(),
        "PIL": mock_pil,
        "cv2": mock_cv2,
        "tflite_runtime": MagicMock(),
        "tflite_runtime.interpreter": mock_tflite_interp,
        "ai_edge_litert": MagicMock(),
        "ai_edge_litert.interpreter": mock_litert_interp,
        "pose_estimator": mock_pose_estimator_module,
    }
    extra_modules["tts_engine"].TTSEngine = mock_tts_class
    extra_modules["session_logger"].SessionLogger = mock_logger_class
    extra_modules["session_chat.app"].start_server = mock_start_server

    with patch.dict(sys.modules, extra_modules):
        import gui

        gui._test_mocks = {
            "tts_class": mock_tts_class,
            "tts_instance": mock_tts_instance,
            "logger_class": mock_logger_class,
            "logger_instance": mock_logger_instance,
            "start_server": mock_start_server,
            "qrcode": mock_qrcode,
            "qr_img": mock_qr_img,
            "pose_estimator_cls": mock_pose_estimator_cls,
            "pose_estimator_inst": mock_pose_estimator_inst,
        }
        yield gui

    if "gui" in sys.modules:
        del sys.modules["gui"]


def test_tts_initialized_on_startup(gui_mod):
    gui_mod._test_mocks["tts_instance"].announce_session_start.assert_called_once()


def test_logger_initialized_on_startup(gui_mod):
    assert gui_mod._test_mocks["logger_class"].call_count == 1


def _run_one_inference_iteration(gui_mod):
    dummy_frame = np.zeros((12, 2), dtype=np.float32)
    for _ in range(30):
        gui_mod.frame_buffer.append(dummy_frame)

    for _ in range(5):
        gui_mod.prediction_buffer.append((0, 1, 0.9))

    gui_mod._classifier_output_queue.put((0, 1, 0.9))

    gui_mod.frame_counter = 4

    gui_mod.tts.update.reset_mock()
    gui_mod.logger.log_frame.reset_mock()

    gui_mod.update()


def test_inference_loop_calls_tts_update(gui_mod):
    _run_one_inference_iteration(gui_mod)
    gui_mod.tts.update.assert_called_once_with("Deep Squat", 1, ANY)


def test_inference_loop_calls_logger_log_frame(gui_mod):
    _run_one_inference_iteration(gui_mod)
    gui_mod.logger.log_frame.assert_called_once()
    args = gui_mod.logger.log_frame.call_args[0]
    assert args[0] == 0
    assert args[1] == 1
    assert args[2] == pytest.approx(0.9)


def test_update_calls_submit_frame(gui_mod):
    _run_one_inference_iteration(gui_mod)
    gui_mod._test_mocks["pose_estimator_inst"].submit_frame.assert_called()


def test_end_session_calls_logger_end_session(gui_mod):
    gui_mod._end_session()
    gui_mod.logger.end_session.assert_called_once()


def test_end_session_calls_start_server(gui_mod):
    gui_mod._end_session()
    gui_mod.start_server.assert_called_once_with(FAKE_SESSION)


def test_end_session_displays_qr_code(gui_mod):
    gui_mod._end_session()
    gui_mod.qrcode.make.assert_called_once()
    url_passed = gui_mod.qrcode.make.call_args[0][0]
    assert ":5000" in url_passed
