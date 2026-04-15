#!/usr/bin/env python3

import sys
import os
import ctypes
import argparse
import sqlite3
import numpy as np

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

sys.path.append('/opt/nvidia/deepstream/deepstream-6.0/lib')
import pyds

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")
SGIE_UNIQUE_ID = 2
EMBEDDING_DIM = 512
PGIE_CONFIDENCE_THRESHOLD = 0.8
MUXER_WIDTH = 640
MUXER_HEIGHT = 480

enrolled = False
pipeline_ref = None
loop_ref = None


def save_embedding(name, embedding):
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    blob = embedding.astype(np.float32).tobytes()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("SELECT id FROM known_faces WHERE name = ?", (name,))
    existing = cursor.fetchone()

    if existing:
        cursor.execute("UPDATE known_faces SET embedding = ? WHERE name = ?", (blob, name))
    else:
        cursor.execute(
            "INSERT INTO known_faces (name, embedding) VALUES (?, ?)",
            (name, blob)
        )

    conn.commit()
    conn.close()


def extract_embedding(obj_meta):
    user_meta_list = obj_meta.obj_user_meta_list
    while user_meta_list is not None:
        try:
            user_meta = pyds.NvDsUserMeta.cast(user_meta_list.data)
        except StopIteration:
            break

        if user_meta.base_meta.meta_type != pyds.NvDsMetaType.NVDSINFER_TENSOR_OUTPUT_META:
            try:
                user_meta_list = user_meta_list.next
            except StopIteration:
                break
            continue

        tensor_meta = pyds.NvDsInferTensorMeta.cast(user_meta.user_meta_data)

        if tensor_meta.unique_id != SGIE_UNIQUE_ID:
            try:
                user_meta_list = user_meta_list.next
            except StopIteration:
                break
            continue

        layer = pyds.get_nvds_LayerInfo(tensor_meta, 0)
        if layer is None:
            try:
                user_meta_list = user_meta_list.next
            except StopIteration:
                break
            continue

        ptr = ctypes.cast(pyds.get_ptr(layer.buffer), ctypes.POINTER(ctypes.c_float))
        embedding = np.ctypeslib.as_array(ptr, shape=(EMBEDDING_DIM,)).copy()
        return embedding

        try:
            user_meta_list = user_meta_list.next
        except StopIteration:
            break

    return None


def sink_pad_buffer_probe(pad, info, user_data):
    global enrolled, pipeline_ref

    if enrolled:
        return Gst.PadProbeReturn.OK

    person_name = user_data

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        return Gst.PadProbeReturn.OK

    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    frame_list = batch_meta.frame_meta_list

    while frame_list is not None:
        try:
            frame_meta = pyds.NvDsFrameMeta.cast(frame_list.data)
        except StopIteration:
            break

        obj_list = frame_meta.obj_meta_list

        while obj_list is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(obj_list.data)
            except StopIteration:
                break

            if obj_meta.confidence < PGIE_CONFIDENCE_THRESHOLD:
                try:
                    obj_list = obj_list.next
                except StopIteration:
                    break
                continue

            embedding = extract_embedding(obj_meta)

            if embedding is not None:
                save_embedding(person_name, embedding)
                enrolled = True
                print("\nSuccessfully enrolled {}!".format(person_name))
                print("Embedding shape: {}".format(embedding.shape))
                print("Embedding L2 norm (pre-normalize): {:.4f}".format(np.linalg.norm(embedding)))

                if pipeline_ref:
                    pipeline_ref.send_event(Gst.Event.new_eos())

                return Gst.PadProbeReturn.OK

            try:
                obj_list = obj_list.next
            except StopIteration:
                break

        try:
            frame_list = frame_list.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def main():
    global pipeline_ref, loop_ref

    parser = argparse.ArgumentParser(description="Enroll a face into the attendance database.")
    parser.add_argument("--name", required=True, help="Name of the person to enroll.")
    parser.add_argument("--cam", default="/dev/video0", help="Camera device path (default: /dev/video0).")
    args = parser.parse_args()

    if not os.path.exists(DB_PATH):
        print("[ERROR] Database not found at {}".format(DB_PATH))
        print("[ERROR] Run 'python3 init_db.py create' first.")
        sys.exit(1)

    print("[INFO] Enrolling: {}".format(args.name))
    print("[INFO] Camera:    {}".format(args.cam))
    print("[INFO] Database:  {}".format(DB_PATH))
    print("[INFO] Waiting for a high-confidence face detection (>{})...".format(PGIE_CONFIDENCE_THRESHOLD))

    Gst.init(None)

    pipeline = Gst.Pipeline()
    pipeline_ref = pipeline

    source = Gst.ElementFactory.make("v4l2src", "camera-source")
    source.set_property("device", args.cam)

    caps1 = Gst.ElementFactory.make("capsfilter", "source-caps")
    caps1.set_property("caps",
        Gst.Caps.from_string("video/x-raw, width=640, height=480, framerate=30/1"))

    vidconv = Gst.ElementFactory.make("videoconvert", "video-convert")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nv-video-convert")

    caps2 = Gst.ElementFactory.make("capsfilter", "nvmm-caps")
    caps2.set_property("caps",
        Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", 1)
    streammux.set_property("width", MUXER_WIDTH)
    streammux.set_property("height", MUXER_HEIGHT)
    streammux.set_property("batched-push-timeout", 40000)
    streammux.set_property("live-source", 1)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "config_infer_primary_yolo.txt")

    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    sgie.set_property("config-file-path", "config_infer_secondary_arcface.txt")

    sink = Gst.ElementFactory.make("fakesink", "output-sink")
    sink.set_property("sync", 0)

    for elem in [source, caps1, vidconv, nvvidconv, caps2, streammux, pgie, sgie, sink]:
        if not elem:
            sys.stderr.write("Failed to create element: {}\n".format(elem))
            sys.exit(1)
        pipeline.add(elem)

    source.link(caps1)
    caps1.link(vidconv)
    vidconv.link(nvvidconv)
    nvvidconv.link(caps2)

    srcpad = caps2.get_static_pad("src")
    sinkpad = streammux.get_request_pad("sink_0")
    srcpad.link(sinkpad)

    streammux.link(pgie)
    pgie.link(sgie)
    sgie.link(sink)

    sink_pad = sink.get_static_pad("sink")
    sink_pad.add_probe(Gst.PadProbeType.BUFFER, sink_pad_buffer_probe, args.name)

    loop = GLib.MainLoop()
    loop_ref = loop

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("[INFO] Pipeline received EOS. Shutting down.")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            sys.stderr.write("ERROR: {} : {}\n".format(err, debug))
            loop.quit()

    bus.connect("message", on_message)

    print("[INFO] Starting enrollment pipeline...")
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    finally:
        pipeline.set_state(Gst.State.NULL)

    if enrolled:
        print("[DONE] {} is now in the database.".format(args.name))
    else:
        print("[WARN] No face was captured. Try again with better lighting.")


if __name__ == "__main__":
    main()
