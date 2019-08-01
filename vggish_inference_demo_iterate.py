# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""A simple demonstration of running VGGish in inference mode.

This is intended as a toy example that demonstrates how the various building
blocks (feature extraction, model definition and loading, postprocessing) work
together in an inference context.

A WAV file (assumed to contain signed 16-bit PCM samples) is read in, converted
into log mel spectrogram examples, fed into VGGish, the raw embedding output is
whitened and quantized, and the postprocessed embeddings are optionally written
in a SequenceExample to a TFRecord file (using the same format as the embedding
features released in AudioSet).

Usage:
  # Run a WAV file through the model and print the embeddings. The model
  # checkpoint is loaded from vggish_model.ckpt and the PCA parameters are
  # loaded from vggish_pca_params.npz in the current directory.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file

  # Run a WAV file through the model and also write the embeddings to
  # a TFRecord file. The model checkpoint and PCA parameters are explicitly
  # passed in as well.
  $ python vggish_inference_demo.py --wav_file /path/to/a/wav/file \
                                    --tfrecord_file /path/to/tfrecord/file \
                                    --checkpoint /path/to/model/checkpoint \
                                    --pca_params /path/to/pca/params

  # Run a built-in input (a sine wav) through the model and print the
  # embeddings. Associated model files are read from the current directory.
  $ python vggish_inference_demo.py
"""

from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os

flags = tf.app.flags

#wavfile_path = "/Users/zoe/Documents/computing-projects/summer-urop/data/FSDKaggle2018.audio_test"
wavfile_path = "./FSDKaggle2018.audio_test"

flags.DEFINE_string(
    'wav_file', wavfile_path,
    'Path to a wav file. Should contain signed 16-bit PCM samples. '
    'If none is provided, a synthetic sound is used.')

flags.DEFINE_string(
    'checkpoint', 'vggish_model.ckpt',
    'Path to the VGGish checkpoint file.')

flags.DEFINE_string(
    'pca_params', 'vggish_pca_params.npz',
    'Path to the VGGish PCA parameters file.')

flags.DEFINE_string(
    'tfrecord_file', None,
    'Path to a TFRecord file where embeddings will be written.')

FLAGS = flags.FLAGS

def main(_):
  # In this simple example, we run the examples from a single audio file through
  # the model. If none is provided, we generate a synthetic input.
  #
  master_embeddings_list = []

  print("begin vggish_inference_demo file")
  if FLAGS.wav_file:
    wav_file = FLAGS.wav_file
  else:
    # Write a WAV of a sine wav into an in-memory file object.
    num_secs = 5
    freq = 1000
    sr = 44100
    t = np.linspace(0, num_secs, int(num_secs * sr))
    x = np.sin(2 * np.pi * freq * t)
    # Convert to signed 16-bit samples.
    samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
    wav_file = six.BytesIO()
    wavfile.write(wav_file, sr, samples)
    wav_file.seek(0)


  print("after initializing wav_file")

  num_secs = 3
  freq = 1000
  sr = 44100
  t = np.linspace(0, num_secs, int(num_secs * sr))
  x = np.sin(2 * np.pi * freq * t)

  # Produce a batch of log mel spectrogram examples.


  # Iterate through directory
  print("began running session")
  # examples_batch = vggish_input.wavfile_to_examples(wav_file)

  # print(examples_batch)
  # input_smoke_test_batch = vggish_input.waveform_to_examples(x, sr)
  # print("Shape of examples batch:", examples_batch.shape, "Shape of smoke test batch: ", input_smoke_test_batch.shape)

  

  with tf.Graph().as_default(), tf.Session() as sess:
      print(" with tf.Graph().as_default(), tf.Session() as sess:")

          # Prepare a postprocessor to munge the model embeddings.
      pproc = vggish_postprocess.Postprocessor(FLAGS.pca_params)

      # If needed, prepare a record writer to store the postprocessed embeddings.
      writer = tf.python_io.TFRecordWriter(
          FLAGS.tfrecord_file) if FLAGS.tfrecord_file else None

      # Define the model in inference mode, load the checkpoint, and
      # locate input and output tensors.
      vggish_slim.define_vggish_slim(training=False)

      print("before load vggish slim")
      vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
      features_tensor = sess.graph.get_tensor_by_name(
          vggish_params.INPUT_TENSOR_NAME)
      embedding_tensor = sess.graph.get_tensor_by_name(
          vggish_params.OUTPUT_TENSOR_NAME)




      for filename in os.listdir(wavfile_path):
          if filename.endswith(".wav"):
              wav_file = os.path.join(wavfile_path, filename)
              print("before_wav_file")
              print("wav_file")
              print(wav_file)
              new_examples_batch = vggish_input.wavfile_to_examples(wav_file)
              print("new_examples_batch")
              print(new_examples_batch)
              [embedding_batch] = sess.run([embedding_tensor],
                                           feed_dict={features_tensor: new_examples_batch})
              print("embedding_batch")
              print(embedding_batch)
              postprocessed_batch = pproc.postprocess(embedding_batch)
              print("postprocessed_batch")
              print(postprocessed_batch)

              master_embeddings_list.append(embedding_batch)

              print(master_embeddings_list)
              continue
          else:
              continue

  # Things to try
  # input noise tensor
  # example audio




# def process_single_file(wav_file):
#
#
#     # Run inference and postprocessing.
#
#
#     return postprocessed_batch
#
#         # Write the postprocessed embeddings as a SequenceExample, in a similar
#         # format as the features released in AudioSet. Each row of the batch of
#         # embeddings corresponds to roughly a second of audio (96 10ms frames), and
#         # the rows are written as a sequence of bytes-valued features, where each
#         # feature value contains the 128 bytes of the whitened quantized embedding.
#         # seq_example = tf.train.SequenceExample(
#         #     feature_lists=tf.train.FeatureLists(
#         #         feature_list={
#         #             vggish_params.AUDIO_EMBEDDING_FEATURE_NAME:
#         #                 tf.train.FeatureList(
#         #                     feature=[
#         #                         tf.train.Feature(
#         #                             bytes_list=tf.train.BytesList(
#         #                                 value=[embedding.tobytes()]))
#         #                         for embedding in postprocessed_batch
#         #                     ]
#         #                 )
#         #         }
#         #     )
#         # )
#         # print(seq_example)
#         # if writer:
#         #     writer.write(seq_example.SerializeToString())
#
#         # if writer:
#         #     writer.close()


if __name__ == '__main__':
  tf.app.run()
