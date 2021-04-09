from __future__ import division
import random
random.seed(1)
import inspect
from collections import namedtuple, defaultdict, OrderedDict
import numpy as np
np.random.seed(1)
from sklearn.model_selection import train_test_split
#from simdna import simulations
import simulations
from simdna.synthetic import StringEmbeddable
from utils import get_motif_scores, one_hot_encode
from models import SequenceDNN
from dragonn.plot import add_letters_to_axis, plot_motif
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def SequenceDNN_learning_curve(dnn):
  if dnn.valid_metrics is not None:
    train_losses, valid_losses = [
        np.array([epoch_metrics['Loss']
                  for epoch_metrics in metrics])
        for metrics in (dnn.train_metrics, dnn.valid_metrics)
    ]
    min_loss_indx = min(enumerate(valid_losses), key=lambda x: x[1])[0]
    f = plt.figure(figsize=(10, 4))
    ax = f.add_subplot(1, 1, 1)
    ax.plot(range(len(train_losses)), train_losses, 'b', label='Training', lw=4)
    ax.plot(
        range(len(train_losses)), valid_losses, 'r', label='Validation', lw=4)
    ax.plot([min_loss_indx, min_loss_indx], [0, 1.0], 'k--', label='Early Stop')
    ax.legend(loc="upper right")
    ax.set_ylabel("Loss")
    ax.set_ylim((0.0, 1.0))
    ax.set_xlabel("Epoch")
    plt.show()
  else:
    print("learning curve can only be obtained after training!")


def test_SequenceDNN(dnn, simulation_data):
  print("Test performance:")
  print(dnn.test(simulation_data.X_test, simulation_data.y_test))


def plot_motifs(simulation_data):
  for motif_name in simulation_data.motif_names:
    plot_motif(motif_name, figsize=(10, 4), ylab=motif_name)


def plot_sequence_filters(dnn):
  fig = plt.figure(figsize=(15, 8))
  fig.subplots_adjust(hspace=0.1, wspace=0.1)
  conv_filters = dnn.get_sequence_filters()
  num_plots_per_axis = int(len(conv_filters)**0.5) + 1
  for i, conv_filter in enumerate(conv_filters):
    ax = fig.add_subplot(num_plots_per_axis, num_plots_per_axis, i + 1)
    add_letters_to_axis(ax, conv_filter.T)
    ax.axis("off")
    ax.set_title("Filter %s" % (str(i + 1)))


def plot_SequenceDNN_layer_outputs(dnn, simulation_data):
  # define layer out functions
  import theano
  get_conv_output = theano.function(
      [dnn.model.layers[0].input],
      dnn.model.layers[0].get_output(train=False),
      allow_input_downcast=True)
  get_conv_relu_output = theano.function(
      [dnn.model.layers[0].input],
      dnn.model.layers[1].get_output(train=False),
      allow_input_downcast=True)
  get_maxpool_output = theano.function(
      [dnn.model.layers[0].input],
      dnn.model.layers[-4].get_output(train=False),
      allow_input_downcast=True)
  # get layer outputs for a positive simulation example
  pos_indx = np.where(simulation_data.y_valid == 1)[0][0]
  pos_X = simulation_data.X_valid[pos_indx:(pos_indx + 1)]
  conv_outputs = get_conv_output(pos_X).squeeze()
  conv_relu_outputs = get_conv_relu_output(pos_X).squeeze()
  maxpool_outputs = get_maxpool_output(pos_X).squeeze()
  # plot layer outputs
  fig = plt.figure(figsize=(15, 12))
  ax1 = fig.add_subplot(3, 1, 3)
  heatmap = ax1.imshow(
      conv_outputs, aspect='auto', interpolation='None', cmap='seismic')
  fig.colorbar(heatmap)
  ax1.set_ylabel("Convolutional Filters")
  ax1.set_xlabel("Position")
  ax1.get_yaxis().set_ticks([])
  ax1.get_xaxis().set_ticks([])
  ax1.set_title("SequenceDNN outputs from convolutional layer.\t\
    Locations of motif sites are highlighted in grey.")

  ax2 = fig.add_subplot(3, 1, 2)
  heatmap = ax2.imshow(
      conv_relu_outputs, aspect='auto', interpolation='None', cmap='seismic')
  fig.colorbar(heatmap)
  ax2.set_ylabel("Convolutional Filters")
  ax2.get_yaxis().set_ticks([])
  ax2.get_xaxis().set_ticks([])
  ax2.set_title("Convolutional outputs after ReLU transformation.\t\
    Locations of motif sites are highlighted in grey.")

  ax3 = fig.add_subplot(3, 1, 1)
  heatmap = ax3.imshow(
      maxpool_outputs, aspect='auto', interpolation='None', cmap='seismic')
  fig.colorbar(heatmap)
  ax3.set_title("DNN outputs after max pooling")
  ax3.set_ylabel("Convolutional Filters")
  ax3.get_yaxis().set_ticks([])
  ax3.get_xaxis().set_ticks([])

  # highlight motif sites
  motif_scores = get_motif_scores(pos_X, simulation_data.motif_names)
  motif_sites = [np.argmax(motif_scores[0, i, :]) for i in [0, 1]]
  for motif_site in motif_sites:
    conv_output_start = motif_site - max(dnn.conv_width - 10, 0)
    conv_output_stop = motif_site + max(dnn.conv_width - 10, 0)
    ax1.axvspan(conv_output_start, conv_output_stop, color='grey', alpha=0.5)
    ax2.axvspan(conv_output_start, conv_output_stop, color='grey', alpha=0.5)


def interpret_SequenceDNN_filters(dnn, simulation_data):
  print("Plotting simulation motifs...")
  plot_motifs(simulation_data)
  plt.show()
  print("Visualizing convolutional sequence filters in SequenceDNN...")
  plot_sequence_filters(dnn)
  plt.show()


def interpret_data_with_SequenceDNN(dnn, simulation_data):
  # get a positive and a negative example from the simulation data
  pos_indx = np.flatnonzero(simulation_data.y_valid == 1)[2]
  neg_indx = np.flatnonzero(simulation_data.y_valid == 0)[2]
  pos_X = simulation_data.X_valid[pos_indx:pos_indx + 1]
  neg_X = simulation_data.X_valid[neg_indx:neg_indx + 1]
  # get motif scores, ISM scores, and DeepLIFT scores
  scores_dict = defaultdict(OrderedDict)
  scores_dict['Positive']['Motif Scores'] = get_motif_scores(
      pos_X, simulation_data.motif_names)
  scores_dict['Positive']['ISM Scores'] = dnn.in_silico_mutagenesis(pos_X).max(
      axis=-2)
  scores_dict['Positive']['DeepLIFT Scores'] = dnn.deeplift(pos_X).max(axis=-2)
  scores_dict['Negative']['Motif Scores'] = get_motif_scores(
      neg_X, simulation_data.motif_names)
  scores_dict['Negative']['ISM Scores'] = dnn.in_silico_mutagenesis(neg_X).max(
      axis=-2)
  scores_dict['Negative']['DeepLIFT Scores'] = dnn.deeplift(neg_X).max(axis=-2)

  # get motif site locations
  motif_sites = {
      key: [
          embedded_motif.startPos + len(embedded_motif.what.string) // 2
          for embedded_motif in (next(
              embedded_motif
              for embedded_motif in simulation_data.valid_embeddings[index]
              if isinstance(embedded_motif.what, StringEmbeddable) and
              motif_name in embedded_motif.what.stringDescription)
                                 for motif_name in simulation_data.motif_names)
      ]
      for key, index in (('Positive', pos_indx), ('Negative', neg_indx))
  }
  # organize legends
  motif_label_dict = {}
  motif_label_dict['Motif Scores'] = simulation_data.motif_names
  if len(simulation_data.motif_names) == dnn.num_tasks:
    motif_label_dict['ISM Scores'] = simulation_data.motif_names
  else:
    motif_label_dict['ISM Scores'] = ['_'.join(simulation_data.motif_names)]
  motif_label_dict['DeepLIFT Scores'] = motif_label_dict['ISM Scores']
  # plot scores and highlight motif site locations
  seq_length = pos_X.shape[-1]
  plots_per_row = 2
  plots_per_column = 3
  ylim_dict = {
      'Motif Scores': (-80, 30),
      'ISM Scores': (-1.5, 3.0),
      'DeepLIFT Scores': (-1.5, 3.0)
  }
  motif_colors = ['b', 'r', 'c', 'm', 'g', 'k', 'y']
  font_size = 12
  num_x_ticks = 5
  highlight_width = 5
  motif_labels_cache = []

  f = plt.figure(figsize=(10, 12))
  f.subplots_adjust(hspace=0.15, wspace=0.15)
  f.set_tight_layout(True)

  for j, key in enumerate(['Positive', 'Negative']):
    for i, (score_type, scores) in enumerate(scores_dict[key].items()):
      ax = f.add_subplot(plots_per_column, plots_per_row,
                         plots_per_row * i + j + 1)
      ax.set_ylim(ylim_dict[score_type])
      ax.set_xlim((0, seq_length))
      ax.set_frame_on(False)
      if j == 0:  # put y axis and ticks only on left side
        xmin, xmax = ax.get_xaxis().get_view_interval()
        ymin, ymax = ax.get_yaxis().get_view_interval()
        ax.add_artist(
            Line2D((xmin, xmin), (ymin, ymax), color='black', linewidth=2))
        ax.get_yaxis().tick_left()
        for tick in ax.yaxis.get_major_ticks():
          tick.label.set_fontsize(font_size / 1.5)
        ax.set_ylabel(score_type)
      if j > 0:  # remove y axes
        ax.get_yaxis().set_visible(False)
      if i < (plots_per_column - 1):  # remove x axes
        ax.get_xaxis().set_visible(False)
      if i == (plots_per_column - 1):  # set x axis and ticks on bottom
        ax.set_xticks(seq_length / num_x_ticks * (np.arange(num_x_ticks + 1)))
        xmin, xmax = ax.get_xaxis().get_view_interval()
        ymin, ymax = ax.get_yaxis().get_view_interval()
        ax.add_artist(
            Line2D((xmin, xmax), (ymin, ymin), color='black', linewidth=2))
        ax.get_xaxis().tick_bottom()
        for tick in ax.xaxis.get_major_ticks():
          tick.label.set_fontsize(font_size / 1.5)
        ax.set_xlabel("Position")
      if j > 0 and i < (plots_per_column - 1):  # remove all axes
        ax.axis('off')

      add_legend = False
      for _i, motif_label in enumerate(motif_label_dict[score_type]):
        if score_type == 'Motif Scores':
          scores_to_plot = scores[0, _i, :]
        else:
          scores_to_plot = scores[0, 0, 0, :]
        if motif_label not in motif_labels_cache:
          motif_labels_cache.append(motif_label)
          add_legend = True
        motif_color = motif_colors[motif_labels_cache.index(motif_label)]
        ax.plot(scores_to_plot, label=motif_label, c=motif_color)
      if add_legend:
        leg = ax.legend(
            loc=[0, 0.85],
            frameon=False,
            fontsize=font_size,
            ncol=3,
            handlelength=-0.5)
        for legobj in leg.legendHandles:
          legobj.set_color('w')
        for _j, text in enumerate(leg.get_texts()):
          text_color = motif_colors[motif_labels_cache.index(
              motif_label_dict[score_type][_j])]
          text.set_color(text_color)
      for motif_site in motif_sites[key]:
        ax.axvspan(
            motif_site - highlight_width,
            motif_site + highlight_width,
            color='grey',
            alpha=0.1)
