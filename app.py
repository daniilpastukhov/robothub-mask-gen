from functools import partial
from pathlib import Path
import re
from typing import List
import time
import numpy as np
import depthai as dai

from robothub_sdk import App, IS_INTERACTIVE, CameraResolution, StreamType, Config

if IS_INTERACTIVE:
    import cv2


def log_softmax(x: np.ndarray):
    x = np.array(x)
    c = x.max()
    logsumexp = np.log(np.exp(x - c).sum())
    return x - c - logsumexp


class MaskDetector(App):
    def on_initialize(self, unused_devices: List[dai.DeviceInfo]):
        self.msgs = {}
        self.config.add_defaults(send_still_picture=False, detect_threshold=0.5)
        self.fps = 30

    def on_configuration(self, old_configuration: Config):
        pass

    def on_setup(self, device):
        device.configure_camera(
            dai.CameraBoardSocket.RGB,
            res=CameraResolution.THE_4_K,
            preview_size=(640, 480)
        )

        # Detect face
        (_, nn_det_out, nn_det_passthrough) = device.create_nn(
            source=device.streams.color_preview,
            blob_path=Path("./det_model.blob"),
            config_path=Path("./det_model.json"),
            input_size=(300, 300)
        )

        (manip, manip_stream) = device.create_image_manipulator()
        manip.initialConfig.setResize(224, 224)
        manip.inputConfig.setWaitForMessage(True)

        self.script = device.create_script(
            script_path=Path("./script.py"),
            inputs={
                'preview': device.streams.color_preview,
                'passthrough': nn_det_passthrough,
                'face_det_in': nn_det_out
            },
            outputs={
                'manip_img': manip.inputImage,
                'manip_cfg': manip.inputConfig
            }
        )

        (_, nn_mask_out, nn_mask_passthrough) = device.create_nn(
            source=manip_stream,
            blob_path=Path("./sbd_mask.blob"),
            config_path=Path("./sbd_mask.json"),
            input_size=(224, 224)
        )

        if IS_INTERACTIVE:
            device.streams.color_preview.consume(partial(self.add_msg, 'color'))
            nn_det_out.consume(partial(self.add_msg, 'detection'))
            nn_mask_out.consume(partial(self.add_msg, 'recognition'))

            streams = (nn_det_out, nn_mask_out, manip_stream)
            device.streams.synchronize(streams, partial(self.on_detection, device.id))
        else:
            encoder = device.create_encoder(
                manip_stream.output_node,
                fps=self.fps,
                profile=dai.VideoEncoderProperties.Profile.MJPEG,
                quality=80,
            )
            encoder_stream = device.streams.create(
                encoder,
                encoder.bitstream,
                stream_type=StreamType.BINARY,
                rate=self.fps,
            )

            device.streams.color_video.publish()
            streams = (nn_det_out, nn_mask_out, encoder_stream)
            device.streams.synchronize(streams, partial(self.on_detection, device.id))

    def on_update(self):
        if not IS_INTERACTIVE:
            return
        
        msgs = self.get_msgs()
        if msgs is None:
            return

        frame = msgs["color"].getCvFrame()
        detections = msgs["detection"].detections
        recognitions = msgs["recognition"]

        is_mask_on = False

        for i, detection in enumerate(detections):
            bbox = self.frame_norm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))

            # Decoding of recognition results
            rec = recognitions[i].getFirstLayerFp16()  # Get NN results. Model only has 1 output layer
            index = np.argmax(log_softmax(rec))
            # Now that we have the classification result we can show it to the user
            text = "No Mask"
            color = (0, 0, 255)
            if index == 1:
                text = "Mask"
                color = (0, 255, 0)
                is_mask_on = True

            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 3)
            y = (bbox[1] + bbox[3]) // 2
            cv2.putText(frame, text, (bbox[0], y), cv2.FONT_HERSHEY_TRIPLEX, 1.5, color, 2)

        if is_mask_on and not IS_INTERACTIVE:
            print('Detected mask on the face')
            self.send_detection(
                f"Mask on",
                frames=[(frame, "jpeg")],
            )

        if IS_INTERACTIVE:
            cv2.imshow("Camera", frame)

        if IS_INTERACTIVE:
            key = cv2.waitKey(1)
            if key == ord("q"):
                self.stop()

    def frame_norm(self, frame, bbox):
        norm_vals = np.full(len(bbox), frame.shape[0])
        norm_vals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * norm_vals).astype(int)

    def add_msg(self, name, msg: dai.NNData):
        seq = str(msg.getSequenceNum())
        if seq not in self.msgs:
            self.msgs[seq] = {}
        if 'recognition' not in self.msgs[seq]:
            self.msgs[seq]['recognition'] = []
        if name == 'recognition':
            self.msgs[seq]['recognition'].append(msg)
        elif name == 'detection':
            self.msgs[seq][name] = msg
            self.msgs[seq]["len"] = len(msg.detections)
        elif name == 'color':
            self.msgs[seq][name] = msg

    def get_msgs(self):
        seq_remove = []
        for seq, msgs in self.msgs.items():
            seq_remove.append(seq)
            if "color" in msgs and "len" in msgs:
                if msgs["len"] == len(msgs["recognition"]):
                    for rm in seq_remove:
                        del self.msgs[rm]
                    return msgs

        return None

    def on_detection(
            self,
            device_id: str,
            obj_data: dai.ImgDetections,
            mask_data: dai.ImgDetections,
            frame: dai.ImgFrame
    ):
        cv_frame = frame.getCvFrame()
        valid_detections = []

        logits = mask_data.getFirstLayerFp16()
        has_mask = np.argmax(log_softmax(logits))

        # Iterate through the object detections
        for detection in obj_data.detections:
            if detection.confidence >= self.config.detect_threshold:
                bbox = self.frame_norm(cv_frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
                valid_detections.append(bbox)

        # Now if you have some valid detections, it should send them
        if len(valid_detections) > 0:
            for valid_bbox in valid_detections:
                if not has_mask:
                    continue

                print(f'Found mask')
                self.send_detection(
                    f'Found mask on frame from device {device_id} with bbox: {valid_bbox}.',
                    tags=['detection'],
                    frames=[(frame, 'jpeg')]
                )


app = MaskDetector()
app.run()
