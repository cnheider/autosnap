# !/usr/bin/env python3
import argparse
import base64
import json
import tkinter as tk
import zlib
from tkinter import BOTH, X
from tkinter.ttk import Frame, Label, LabelFrame

import numpy as np
import pyaudio
import requests
import cv2
import dlib
from _dlib_pybind11 import rectangle
from draugr.opencv_utilities import AsyncVideoStream
from draugr.opencv_utilities.dlib.facealigner import align_face
from draugr.opencv_utilities.dlib_utilities import dlib68FacialLandmarksIndices, eye_aspect_ratio, mouth_aspect_ratio, shape_to_ndarray

LABELS_WTF = ("silence", "unknown", "yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go")
UNCERTAINTY_THRESHOLD = 0.5
detector = dlib.get_frontal_face_detector()
crude_predictor = dlib.shape_predictor("shape_predictor_5_face_landmarks.dat")
detail_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
outer_upsample_num_times = 1
inner_upsample_num_times = 0
debug = True
face_size = (256, 256)
rect_aligned = rectangle(30, 30, 256 - 30, 256 - 30)
stream = iter(AsyncVideoStream())

class SpeechDemo(Frame):
  skeet = LABELS_WTF

  def __init__(self, label_client):
    super().__init__()
    self.label_client = label_client
    self.init_ui()

  def init_ui(self,
              rows=4,
              cols=3):
    """ setup the GUI for the app """
    self.master.title("Speech Demo")
    self.pack(fill=BOTH, expand=True)

    label_frame = LabelFrame(self, text="Try these categories")
    label_frame.pack(fill=X, padx=10, pady=10)
    self.labels = []

    for j in range(rows):
      for i in range(cols):
        k = i + j * cols
        label = Label(label_frame, text=self.skeet[k])
        label.config(font=("Courier", 36))
        label.grid(row=j, column=i, padx=10, pady=10)
        self.labels += [label]
    self.selected = None
    self.after(100, self.on_tick)

  def on_tick(self):
    '''check for new labels and display them'''
    words = self.label_client.get_words()
    if len(words) > 0:
      key = words[-1]
      i = self.skeet.index(key)
      label = self.labels[i]



      if label["text"] != key:
        print(f"That's weird label {i} has text {label['text']}")

      if label != self.selected and self.selected is not None:
        self.selected.configure(background='')

      label.configure(background="green")
      self.selected = label


    frame = next(stream)
    gray_o = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for rect in detector(gray_o, outer_upsample_num_times):
      aligned = align_face(gray_o, gray_o, rect, crude_predictor, desired_face_size=face_size)
      # for rect_aligned in detector(aligned, inner_upsample_num_times): # alternative to hardcoded

      aligned_landmarks = shape_to_ndarray(detail_predictor(aligned, rect_aligned))

      mouth = dlib68FacialLandmarksIndices.slice(aligned_landmarks, dlib68FacialLandmarksIndices.mouth)
      mouth_ar = mouth_aspect_ratio(mouth)

      left_eye = dlib68FacialLandmarksIndices.slice(aligned_landmarks, dlib68FacialLandmarksIndices.left_eye)
      left_eye_ar = eye_aspect_ratio(left_eye)

      right_eye = dlib68FacialLandmarksIndices.slice(aligned_landmarks, dlib68FacialLandmarksIndices.right_eye)
      right_eye_ar = eye_aspect_ratio(right_eye)

      #visual_predictors = [mouth_ar, left_eye_ar, right_eye_ar]
      if self.selected and (self.selected["text"] == 'right' or self.selected["text"] == 'yes'):
        if (left_eye_ar > 0.24 and
            right_eye_ar > 0.24 and
            (mouth_ar <.2 or .34 < mouth_ar < .64)

        ):
            cv2.imwrite(f"smile.png", frame)
            print('capture!!')

      if debug:
        cv2.drawContours(aligned, [cv2.convexHull(mouth)], -1, (0, 255, 0), 1)
        cv2.drawContours(aligned, [cv2.convexHull(left_eye)], -1, (0, 255, 0), 1)
        cv2.drawContours(aligned, [cv2.convexHull(right_eye)], -1, (0, 255, 0), 1)

        cv2.imshow("rect", aligned)

        cv2.putText(frame,
                    f"mouth_AR: {mouth_ar:.2f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2)
        cv2.putText(frame,
                    f"Left eye_AR: {left_eye_ar:.2f}",
                    (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)
        cv2.putText(frame,
                    f"Right eye_AR: {right_eye_ar:.2f}",
                    (300, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2)
        cv2.imshow("test", frame)

    self.after(100, self.on_tick)


class CategoryClient(object):
  def __init__(self, server_endpoint):
    self.endpoint = server_endpoint
    self.chunk_size = 1000
    self._audio = pyaudio.PyAudio()
    self._audio.open(format=pyaudio.paInt16,
                     channels=1,
                     rate=16000,
                     input=True,
                     frames_per_buffer=self.chunk_size,
                     stream_callback=self._on_audio)
    self.last_data = np.zeros(1000)
    self._audio_buf = []
    self.words = []

  def _on_audio(self, in_data, frame_count, time_info, status):
    '''

    :param in_data:
    :param frame_count:
    :param time_info:
    :param status:
    :return:
    '''
    data_ok = (in_data, pyaudio.paContinue)
    self.last_data = in_data
    self._audio_buf.append(in_data)
    if len(self._audio_buf) != 16:
      return data_ok
    audio_data = base64.b64encode(zlib.compress(b"".join(self._audio_buf)))
    self._audio_buf = []
    response = requests.post(f"{self.endpoint}/listen", json=dict(wav_data=audio_data.decode(), method="all_label"))
    response = json.loads(response.content.decode())
    if not response:
      return data_ok

    max_key = max(response.items(), key=lambda x:x[1])[0]
    for key in response:
      p = response[key]
      if p < UNCERTAINTY_THRESHOLD and key != "__unknown__":
        print('try again')
        continue
      key = key.replace("_", "")
      print(key)
      self.words += [key]
    return data_ok

  def get_words(self):
    '''

    :return:
    '''
    temp = self.words
    self.words = []
    return temp


def main(server_endpoint):
  """ Main function to create root UI and SpeechDemo object, then run the main UI loop """
  root = tk.Tk()
  root.geometry("800x600")
  app = SpeechDemo(CategoryClient(server_endpoint))
  while True:
    try:
      root.mainloop()
      break
    except UnicodeDecodeError:
      pass







if __name__ == "__main__":

  cv2.namedWindow("test")
  cv2.namedWindow("rect")

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--server-endpoint",
      type=str,
      default="http://127.0.0.1:16888",
      help="The endpoint to use")
  flags = parser.parse_args()
  main(flags.server_endpoint)



  # do a bit of cleanup
  cv2.destroyAllWindows()