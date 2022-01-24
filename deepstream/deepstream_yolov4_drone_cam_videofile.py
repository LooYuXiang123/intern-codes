#!/usr/bin/env python3

################################################################################
# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
################################################################################

import sys
sys.path.append('../')
import gi
gi.require_version('Gst', '1.0')
from gi.repository import GObject, Gst, GLib
from common.is_aarch_64 import is_aarch64
from common.bus_call import bus_call
from common.FPS import GETFPS

import pyds
import os
import re

#PGIE_CLASS_ID_VEHICLE = 0
#PGIE_CLASS_ID_BICYCLE = 1
#PGIE_CLASS_ID_PERSON = 2
#PGIE_CLASS_ID_ROADSIGN = 3

PGIE_CLASS_ID_DRONE = 0

fps_streams = {}

def osd_sink_pad_buffer_probe(pad, info, u_data):
    prediction_list = []
    frame_number=0
    #Intiallizing object counter with 0.
    obj_counter = {
        #PGIE_CLASS_ID_VEHICLE:0,
        #PGIE_CLASS_ID_PERSON:0,
        #PGIE_CLASS_ID_BICYCLE:0,
        #PGIE_CLASS_ID_ROADSIGN:0
        PGIE_CLASS_ID_DRONE:0
    }
    num_rects=0

    gst_buffer = info.get_buffer()
    if not gst_buffer:
        print("Unable to get GstBuffer ")
        return

    # Retrieve batch metadata from the gst_buffer
    # Note that pyds.gst_buffer_get_nvds_batch_meta() expects the
    # C address of gst_buffer as input, which is obtained with hash(gst_buffer)
    batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
    l_frame = batch_meta.frame_meta_list
    while l_frame is not None:
        try:
            # Note that l_frame.data needs a cast to pyds.NvDsFrameMeta
            # The casting is done by pyds.NvDsFrameMeta.cast()
            # The casting also keeps ownership of the underlying memory
            # in the C code, so the Python garbage collector will leave
            # it alone.
           frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
        except StopIteration:
            break

        frame_number=frame_meta.frame_num
        num_rects = frame_meta.num_obj_meta
        l_obj=frame_meta.obj_meta_list
        if l_obj is None:
            info = [int(frame_number + 1)]
            string = ' '.join([str(elem) for elem in info])
            prediction_list.append(string)
            print('Frame Number:', frame_number + 1)
        while l_obj is not None:
            try:
                # Casting l_obj.data to pyds.NvDsObjectMeta
                obj_meta=pyds.NvDsObjectMeta.cast(l_obj.data)
                #bbox stuff
                top_pos=obj_meta.rect_params.top
                left_pos=obj_meta.rect_params.left
                width_params=obj_meta.rect_params.width
                height_params=obj_meta.rect_params.height
                bottom_pos = top_pos + height_params
                right_pos = left_pos + width_params
                confidence_level = obj_meta.confidence
                #writing bbox to file stuff
                info = [int(frame_number + 1), float(left_pos), float(top_pos), float(right_pos), float(bottom_pos), float(confidence_level)]
                string = ' '.join([str(elem) for elem in info])
                prediction_list.append(string)
                #print bbox stuff
                print('Frame Number:', frame_number + 1)
                print("xmin:",left_pos)
                print("ymin:",top_pos)
                print("xmax:", right_pos)
                print("ymax:", bottom_pos)
                #print("width:",width_params)
                #print("height:",height_params)
                print("Confidence Level:",confidence_level)
            except StopIteration:
                break
            obj_counter[obj_meta.class_id] += 1
            try: 
                l_obj=l_obj.next
            except StopIteration:
                break

        # Acquiring a display meta object. The memory ownership remains in
        # the C code so downstream plugins can still access it. Otherwise
        # the garbage collector will claim it when this probe function exits.
        display_meta=pyds.nvds_acquire_display_meta_from_pool(batch_meta)
        display_meta.num_labels = 1
        py_nvosd_text_params = display_meta.text_params[0]
        # Setting display text to be shown on screen
        # Note that the pyds module allocates a buffer for the string, and the
        # memory will not be claimed by the garbage collector.
        # Reading the display_text field here will return the C address of the
        # allocated string. Use pyds.get_string() to get the string content.
        py_nvosd_text_params.display_text = "Frame Number={} Number of Objects={} Drone_count={}".format(frame_number + 1, num_rects, obj_counter[PGIE_CLASS_ID_DRONE])

        # Now set the offsets where the string should appear
        py_nvosd_text_params.x_offset = 10
        py_nvosd_text_params.y_offset = 12

        # Font , font-color and font-size
        py_nvosd_text_params.font_params.font_name = "Serif"
        py_nvosd_text_params.font_params.font_size = 10
        # set(red, green, blue, alpha); set to White
        py_nvosd_text_params.font_params.font_color.set(1.0, 1.0, 1.0, 1.0)

        # Text background color
        py_nvosd_text_params.set_bg_clr = 1
        # set(red, green, blue, alpha); set to Black
        py_nvosd_text_params.text_bg_clr.set(0.0, 0.0, 0.0, 1.0)
        # Using pyds.get_string() to get display_text as string
        print(pyds.get_string(py_nvosd_text_params.display_text))
        pyds.nvds_add_display_meta_to_frame(frame_meta, display_meta)

        #FPS stuff
        #fps_stream=GETFPS(0)
        #FPS = fps_stream.get_fps() #I have only one source stream, indexed as 0.
        fps_streams["stream{0}".format(frame_meta.pad_index)].get_fps()
        
        try:
            l_frame=l_frame.next
        except StopIteration:
            break
    #writing bbox to file stuff
    #with open('prediction_file.txt', 'a+') as pred_f:     
    with open(os.path.join(folder_name, video + '.txt'), 'a+') as pred_f:
        for elements in prediction_list:
            pred_f.write(elements + '\n')
    pred_f.close()
    return Gst.PadProbeReturn.OK	

def stop_pipeline(pipeline):
    #Eos Signal
    print("Send EoS")
    Gst.Element.send_event(pipeline, Gst.Event.new_eos())
    return True

def main(args):

    # Check input arguments
    if len(args) != 2:
        raise AssertionError('2 Arguments is required')
    
    file_path = args[-1]
    if os.path.exists(file_path + '.mp4'):
        raise AssertionError('Video file already exists')

    global folder_name
    folder_name = os.path.dirname(file_path)
    global video
    video = os.path.basename(file_path)
    #creating folder
    if os.path.exists(folder_name):
        print("The output folder %s already exists.\n" % folder_name)
    else:
        os.mkdir(folder_name)
        print("Creating output folder %s.\n" % folder_name)
    print("Video will be saved in %s.\n" % folder_name)

    #FPS stuff
    fps_streams["stream{0}".format(0)]=GETFPS(0)

    # Standard GStreamer initialization
    GObject.threads_init()
    Gst.init(None)

    # Create gstreamer elements
    # Create Pipeline element that will form a connection of other elements
    print("Creating Pipeline \n ")
    pipeline = Gst.Pipeline()

    if not pipeline:
        sys.stderr.write(" Unable to create Pipeline \n")

    # Source element for reading from the file
    #CSI camera source
    print("Creating Source \n ")
    source = Gst.ElementFactory.make("nvarguscamerasrc", "src-elem")
    if not source:
        sys.stderr.write(" Unable to create Source \n")

    # Converter to scale the image
    #onboard camera stuff i think?
    nvvidconv_src = Gst.ElementFactory.make("nvvideoconvert", "convertor_src")
    if not nvvidconv_src:
        sys.stderr.write(" Unable to create nvvidconv_src \n")

    # Caps for NVMM and resolution scaling
    #onboard camera stuff i think?
    caps_nvvidconv_src = Gst.ElementFactory.make("capsfilter", "nvmm_caps")
    if not caps_nvvidconv_src:
        sys.stderr.write(" Unable to create capsfilter \n")

    # Create nvstreammux instance to form batches from one or more sources.
    streammux = Gst.ElementFactory.make("nvstreammux", "Stream-muxer")
    if not streammux:
        sys.stderr.write(" Unable to create NvStreamMux \n")

    # Use nvinfer to run inferencing on decoder's output,
    # behaviour of inferencing is set through config file
    pgie = Gst.ElementFactory.make("nvinfer", "primary-inference")
    if not pgie:
        sys.stderr.write(" Unable to create pgie \n")

    nvvidconv = Gst.ElementFactory.make("nvvideoconvert", "convertor")
    if not nvvidconv:
        sys.stderr.write(" Unable to create nvvidconv \n")

    # Create OSD to draw on the converted RGBA buffer
    nvosd = Gst.ElementFactory.make("nvdsosd", "onscreendisplay")

    if not nvosd:
        sys.stderr.write(" Unable to create nvosd \n")

    # Finally render the osd output
    #if is_aarch64():
        #transform = Gst.ElementFactory.make("nvegltransform", "nvegl-transform")
    ###
    print("Creating Queue \n")
    queue = Gst.ElementFactory.make("queue", "queue")
    if not queue:
        sys.stderr.write(" Unable to create queue \n")

    print("Creating converter 2\n")
    nvvidconv2 = Gst.ElementFactory.make("nvvideoconvert", "convertor2")
    if not nvvidconv2:
        sys.stderr.write(" Unable to create nvvidconv2 \n")

    print("Creating capsfilter \n")
    capsfilter = Gst.ElementFactory.make("capsfilter", "capsfilter")
    if not capsfilter:
        sys.stderr.write(" Unable to create capsfilter \n")

    caps = Gst.Caps.from_string("video/x-raw, format=I420")
    capsfilter.set_property("caps", caps)

    print("Creating Encoder \n")
    encoder = Gst.ElementFactory.make("avenc_mpeg4", "encoder")
    if not encoder:
        sys.stderr.write(" Unable to create encoder \n")

    encoder.set_property("bitrate", 2000000)

    print("Creating Code Parser \n")
    codeparser = Gst.ElementFactory.make("mpeg4videoparse", "mpeg4-parser")
    if not codeparser:
        sys.stderr.write(" Unable to create code parser \n")

    print("Creating Container \n")
    container = Gst.ElementFactory.make("qtmux", "qtmux")
    if not container:
        sys.stderr.write(" Unable to create code parser \n")

    ### File Sink
    print("Creating Sink \n")
    sink = Gst.ElementFactory.make("filesink", "filesink")
    if not sink:
        sys.stderr.write(" Unable to create file sink \n")
    
    sink.set_property("location", file_path + '.mp4')
    sink.set_property("sync", 1)
    sink.set_property("async", 0)
    ###

    source.set_property('bufapi-version', True)

    caps_nvvidconv_src.set_property('caps', Gst.Caps.from_string('video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1'))

    streammux.set_property('width', 1280)
    streammux.set_property('height', 720)
    streammux.set_property('batch-size', 1)
    streammux.set_property('batched-push-timeout', 1/30)
    streammux.set_property('live-source', 1)
    pgie.set_property('config-file-path', "pgie_yolov4_tlt_config_95.txt")

    print("Adding elements to Pipeline \n")
    pipeline.add(source)
    pipeline.add(nvvidconv_src)
    pipeline.add(caps_nvvidconv_src)
    pipeline.add(streammux)
    pipeline.add(pgie)
    pipeline.add(nvvidconv)
    pipeline.add(nvosd)
    ###
    pipeline.add(queue)
    pipeline.add(nvvidconv2)
    pipeline.add(capsfilter)
    pipeline.add(encoder)
    pipeline.add(codeparser)
    pipeline.add(container)
    pipeline.add(sink)
    ###

    # we link the elements together
    print("Linking elements in the Pipeline \n")
    #source
    source.link(nvvidconv_src)
    nvvidconv_src.link(caps_nvvidconv_src)

    sinkpad = streammux.get_request_pad("sink_0")
    if not sinkpad:
        sys.stderr.write(" Unable to get the sink pad of streammux \n")
    srcpad = caps_nvvidconv_src.get_static_pad("src")
    if not srcpad:
        sys.stderr.write(" Unable to get source pad of source \n")
    srcpad.link(sinkpad)
    streammux.link(pgie)
    pgie.link(nvvidconv)
    nvosd.set_property('display-text',0)
    nvvidconv.link(nvosd)
    ###
    nvosd.link(queue)
    queue.link(nvvidconv2)
    nvvidconv2.link(capsfilter)
    capsfilter.link(encoder)
    encoder.link(codeparser)
    codeparser.link(container)
    container.link(sink)
    ###

    timeout = 30 * 1000
    # create and event loop and feed gstreamer bus mesages to it
    loop = GObject.MainLoop()
    bus = pipeline.get_bus()
    bus.add_signal_watch()
    bus.connect ("message", bus_call, loop)

    # Lets add probe to get informed of the meta data generated, we add probe to
    # the sink pad of the osd element, since by that time, the buffer would have
    # had got all the metadata.
    osdsinkpad = nvosd.get_static_pad("sink")
    if not osdsinkpad:
        sys.stderr.write(" Unable to get sink pad of nvosd \n")
    osdsinkpad.add_probe(Gst.PadProbeType.BUFFER, osd_sink_pad_buffer_probe, 0)

    # start play back and listed to events
    print("Starting pipeline \n")
    pipeline.set_state(Gst.State.PLAYING)
    GLib.timeout_add(timeout, stop_pipeline, pipeline)
    try:
        loop.run()
    except:
        pass
    
    # cleanup
    pipeline.set_state(Gst.State.NULL)

if __name__ == '__main__':
    sys.exit(main(sys.argv))

