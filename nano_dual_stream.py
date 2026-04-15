#!/usr/bin/env python3

import sys
import os
import ctypes
import threading
import time
import sqlite3
import numpy as np
from queue import Queue, Empty

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib

sys.path.append('/opt/nvidia/deepstream/deepstream-6.0/lib')
import pyds

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "attendance.db")
MATCH_THRESHOLD = 0.6
COOLDOWN_SECONDS = 60
SGIE_UNIQUE_ID = 2
EMBEDDING_DIM = 512
MUXER_WIDTH = 640
MUXER_HEIGHT = 480
MUXER_BATCH_TIMEOUT = 40000

INPUT_SOURCES = [
    "v4l2:///dev/video0",
    "v4l2:///dev/video1",
]

attendance_queue = Queue()
logged_tracker_ids = {}
known_faces_cache = []


def load_known_faces():
    global known_faces_cache
    if not os.path.exists(DB_PATH):
        print("[WARN] Database not found at {}. Run init_db.py first.".format(DB_PATH))
        return
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT name, embedding FROM known_faces")
    rows = cursor.fetchall()
    conn.close()
    known_faces_cache = []
    for name, blob in rows:
        emb = np.frombuffer(blob, dtype=np.float32).copy()
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        known_faces_cache.append((name, emb))
    print("[INFO] Loaded {} known faces into cache.".format(len(known_faces_cache)))


def cosine_similarity(a, b):
    dot = np.dot(a, b)
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(dot / (na * nb))


def match_embedding(embedding):
    best_name = None
    best_score = -1.0
    for name, known_emb in known_faces_cache:
        score = cosine_similarity(embedding, known_emb)
        if score > best_score:
            best_score = score
            best_name = name
    if best_score >= MATCH_THRESHOLD:
        return best_name, best_score
    return None, best_score


def db_writer_thread():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    while True:
        try:
            record = attendance_queue.get(timeout=2.0)
        except Empty:
            continue
        if record is None:
            break
        person_name, camera_id, tracker_id, confidence = record
        cursor.execute(
            "INSERT INTO attendance_logs (person_name, camera_id, tracker_id, confidence) VALUES (?, ?, ?, ?)",
            (person_name, camera_id, tracker_id, round(confidence, 4))
        )
        conn.commit()
        print("[ATTENDANCE] {} | cam={} trk={} conf={:.4f}".format(
            person_name, camera_id, tracker_id, confidence))
    conn.close()
    print("[INFO] DB writer thread exiting.")


def extract_sgie_embedding(obj_meta):
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


def osd_sink_pad_buffer_probe(pad, info, u_data):
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

        source_id = frame_meta.source_id
        obj_list = frame_meta.obj_meta_list

        while obj_list is not None:
            try:
                obj_meta = pyds.NvDsObjectMeta.cast(obj_list.data)
            except StopIteration:
                break

            tracker_id = obj_meta.object_id

            embedding = extract_sgie_embedding(obj_meta)

            if embedding is not None:
                person_name, confidence = match_embedding(embedding)

                if person_name is not None:
                    cache_key = (source_id, tracker_id)
                    now = time.time()

                    if cache_key in logged_tracker_ids:
                        last_time = logged_tracker_ids[cache_key]
                        if (now - last_time) < COOLDOWN_SECONDS:
                            try:
                                obj_list = obj_list.next
                            except StopIteration:
                                break
                            continue

                    logged_tracker_ids[cache_key] = now
                    attendance_queue.put((person_name, source_id, tracker_id, confidence))

            try:
                obj_list = obj_list.next
            except StopIteration:
                break

        try:
            frame_list = frame_list.next
        except StopIteration:
            break

    return Gst.PadProbeReturn.OK


def create_source_bin(index, uri):
    bin_name = "source-bin-{:02d}".format(index)

    if uri.startswith("v4l2://"):
        device = uri.replace("v4l2://", "")
        src = Gst.ElementFactory.make("v4l2src", "src-{}".format(index))
        src.set_property("device", device)
        caps = Gst.ElementFactory.make("capsfilter", "caps-{}".format(index))
        caps.set_property("caps",
            Gst.Caps.from_string("video/x-raw, width=640, height=480, framerate=30/1"))
        vidconv = Gst.ElementFactory.make("videoconvert", "vidconv-{}".format(index))
        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-{}".format(index))
        capsfilter2 = Gst.ElementFactory.make("capsfilter", "caps2-{}".format(index))
        capsfilter2.set_property("caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

        nbin = Gst.Bin.new(bin_name)
        nbin.add(src)
        nbin.add(caps)
        nbin.add(vidconv)
        nbin.add(nvvidconv)
        nbin.add(capsfilter2)
        src.link(caps)
        caps.link(vidconv)
        vidconv.link(nvvidconv)
        nvvidconv.link(capsfilter2)

        pad = capsfilter2.get_static_pad("src")
        ghost = Gst.GhostPad.new("src", pad)
        nbin.add_pad(ghost)
        return nbin

    else:
        nbin = Gst.Bin.new(bin_name)
        uri_decode = Gst.ElementFactory.make("uridecodebin", "uri-decode-{}".format(index))
        uri_decode.set_property("uri", uri)
        uri_decode.connect("pad-added", _on_pad_added, nbin, index)

        nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "nvvidconv-{}".format(index))
        capsfilter = Gst.ElementFactory.make("capsfilter", "caps-{}".format(index))
        capsfilter.set_property("caps",
            Gst.Caps.from_string("video/x-raw(memory:NVMM), format=NV12"))

        nbin.add(uri_decode)
        nbin.add(nvvidconv)
        nbin.add(capsfilter)
        nvvidconv.link(capsfilter)

        pad = capsfilter.get_static_pad("src")
        ghost = Gst.GhostPad.new("src", pad)
        ghost.set_active(True)
        nbin.add_pad(ghost)
        return nbin


def _on_pad_added(src, new_pad, nbin, index):
    caps = new_pad.get_current_caps()
    struct = caps.get_structure(0)
    media_type = struct.get_name()
    if media_type.startswith("video"):
        nvvidconv = nbin.get_by_name("nvvidconv-{}".format(index))
        sink_pad = nvvidconv.get_static_pad("sink")
        if not sink_pad.is_linked():
            new_pad.link(sink_pad)


def main():
    load_known_faces()

    Gst.init(None)

    pipeline = Gst.Pipeline()
    if not pipeline:
        sys.stderr.write("Failed to create pipeline\n")
        sys.exit(1)

    streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
    streammux.set_property("batch-size", len(INPUT_SOURCES))
    streammux.set_property("width", MUXER_WIDTH)
    streammux.set_property("height", MUXER_HEIGHT)
    streammux.set_property("batched-push-timeout", MUXER_BATCH_TIMEOUT)
    streammux.set_property("live-source", 1)
    pipeline.add(streammux)

    for i, uri in enumerate(INPUT_SOURCES):
        source_bin = create_source_bin(i, uri)
        pipeline.add(source_bin)
        srcpad = source_bin.get_static_pad("src")
        sinkpad = streammux.get_request_pad("sink_{}".format(i))
        srcpad.link(sinkpad)

    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    pgie.set_property("config-file-path", "config_infer_primary_yolo.txt")
    pipeline.add(pgie)

    tracker = Gst.ElementFactory.make("nvtracker", "tracker")
    tracker.set_property("tracker-width", 640)
    tracker.set_property("tracker-height", 480)
    tracker.set_property("ll-lib-file",
        "/opt/nvidia/deepstream/deepstream-6.0/lib/libnvds_nvmultiobjecttracker.so")
    tracker.set_property("ll-config-file", "tracker_config.yml")
    tracker.set_property("gpu-id", 0)
    pipeline.add(tracker)

    sgie = Gst.ElementFactory.make("nvinfer", "secondary-inference")
    sgie.set_property("config-file-path", "config_infer_secondary_arcface.txt")
    pipeline.add(sgie)

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    pipeline.add(nvvidconv)

    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")
    nvosd.set_property("process-mode", 2)
    pipeline.add(nvosd)

    if os.environ.get("DISPLAY"):
        print("[INFO] Display detected, using nv3dsink.")
        sink = Gst.ElementFactory.make("nv3dsink", "sink")
        sink.set_property("sync", 0)
    else:
        print("[INFO] No display, using fakesink.")
        sink = Gst.ElementFactory.make("fakesink", "sink")
        sink.set_property("sync", 0)
    pipeline.add(sink)

    streammux.link(pgie)
    pgie.link(tracker)
    tracker.link(sgie)
    sgie.link(nvvidconv)
    nvvidconv.link(nvosd)
    nvosd.link(sink)

    osdsinkpad = nvosd.get_static_pad("sink")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    db_thread = threading.Thread(target=db_writer_thread, daemon=True)
    db_thread.start()

    loop = GLib.MainLoop()

    bus = pipeline.get_bus()
    bus.add_signal_watch()

    def on_message(bus, msg):
        t = msg.type
        if t == Gst.MessageType.EOS:
            print("[INFO] End-of-stream.")
            loop.quit()
        elif t == Gst.MessageType.ERROR:
            err, debug = msg.parse_error()
            sys.stderr.write("ERROR: {} : {}\n".format(err, debug))
            loop.quit()
        elif t == Gst.MessageType.WARNING:
            err, debug = msg.parse_warning()
            sys.stderr.write("WARN: {} : {}\n".format(err, debug))

    bus.connect("message", on_message)

    print("[INFO] Starting pipeline with {} sources...".format(len(INPUT_SOURCES)))
    pipeline.set_state(Gst.State.PLAYING)

    try:
        loop.run()
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted. Shutting down...")
    finally:
        pipeline.set_state(Gst.State.NULL)
        attendance_queue.put(None)
        db_thread.join(timeout=5)
        print("[INFO] Pipeline stopped.")


if __name__ == "__main__":
    main()
