import matplotlib.pyplot as plt

def plot_curvature_vs_repetitions(results, sequence_position, offset=0, 
                                  show=True, save_fig=False, save_path=None):
  x = list(range(len(results[sequence_position]["curvatures"])))
  y = results[sequence_position]["curvatures"]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Global Curvature of the Residual Stream")
  plt.ylabel("Global Curvature")
  plt.xlabel("Repetitions")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()

def plot_earliest_1_rr_vs_repetitions(results, sequence_position, offset=0,
                                      show=True, save_fig=False, save_path=None):
  rr_dict = results[sequence_position]["ed_reciprocal_ranks"]
  x = list(range(len(rr_dict[0])))
  earliest_rr_of_1 = []
  for i in range(len(x)):
    early_rr = 12 # default value if no rr of 1
    for layer in range(len(rr_dict)):
      if rr_dict[layer][i] == 1:
        early_rr = layer
        break
    earliest_rr_of_1.append(early_rr)
  y = earliest_rr_of_1
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Earliest RR of 1 Layer vs Repetitions")
  plt.ylabel("Layer")
  plt.xlabel("Repetitions")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()

def plot_rr_vs_repetitions(results, sequence_position, offset=0, layer=11, 
                           show=True, save_fig=False, save_path=None):
  rr_dict = results[sequence_position]["ed_reciprocal_ranks"]
  x = list(range(len(rr_dict[layer])))
  y = rr_dict[layer]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title(f"RR of Layer {layer} vs Repetitions")
  plt.ylabel("RR")
  plt.xlabel("Repetitions")
  if save_fig and save_path:
    print('saving figure')
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()

def plot_loss_vs_repetitions(results, sequence_position, offset=0, 
                             show=True, save_fig=False, save_path=None):
  x = list(range(len(results[sequence_position]["losses"])))
  y = results[sequence_position]["losses"]
  plt.scatter(x[offset:], y[offset:], c="green", marker="^")
  plt.legend()
  plt.title("Loss vs Repetitions")
  plt.ylabel("Loss")
  plt.xlabel("Repetitions")
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()
    
##### USEFUL FXNS FOR FIRST PASS CAPITALS #####

def plot_curvature_vs_icl_examples(dataset_curvatures, show=True, save_fig=False, save_path=None):
  '''Plots the global curvature of the residual stream as a function of the number of ICL examples.'''
  curve_lang0 = dataset_curvatures[0]
  curve_lang1 = dataset_curvatures[1]
  curve_lang2 = dataset_curvatures[2]
  curve_lang3 = dataset_curvatures[3]

  curve_c_lang1 = dataset_curvatures[4]
  curve_c_lang2 = dataset_curvatures[5]
  curve_c_lang3 = dataset_curvatures[6]

  curve_an0 = dataset_curvatures[7]
  curve_an1 = dataset_curvatures[8]
  curve_an2 = dataset_curvatures[9]
  curve_an3 = dataset_curvatures[10]

  curve_c_an1 = dataset_curvatures[11]
  curve_c_an2 = dataset_curvatures[12]
  curve_c_an3 = dataset_curvatures[13]

  x = [0] * len(curve_lang0) + [1] * len(curve_lang1) + [2] * len(curve_lang2) + [3] * len(curve_lang3)
  x_counter = x[len(curve_lang0):]

  plt.scatter(x, curve_lang0 + curve_lang1 + curve_lang2 + curve_lang3, c="green", marker="^", label="Language")
  plt.scatter(x_counter, curve_c_lang1 + curve_c_lang2 + curve_c_lang3, c="blue", marker="^", label="Counterfactual Language")

  plt.scatter(x, curve_an0 + curve_an1 + curve_an2 + curve_an3, c="red", marker="o", label="Analogy")
  plt.scatter(x_counter, curve_c_an1 + curve_c_an2 + curve_c_an3, c="orange", marker="o", label="Counterfactual Analogy")

  plt.legend()
  plt.title("Global Curvature of the Residual Stream")
  plt.ylabel("Global Curvature")
  plt.xlabel("Number of ICL Examples")
  plt.xticks([0, 1, 2, 3])
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()
    
def plot_global_curvature_dissected(dataset_curvatures, dataset_mlp_curvatures, dataset_attn_curvatures,
                                    show=True, save_fig=False, save_path=None):
  curve_lang0 = dataset_curvatures[0]
  curve_lang1 = dataset_curvatures[1]
  curve_lang2 = dataset_curvatures[2]
  curve_lang3 = dataset_curvatures[3]
  
  x = [0] * len(curve_lang0) + [1] * len(curve_lang1) + [2] * len(curve_lang2) + [3] * len(curve_lang3)

  plt.scatter(x, curve_lang0 + curve_lang1 + curve_lang2 + curve_lang3, c="green", marker="^", label="Full Model, Language")

  curve_lang0 = dataset_mlp_curvatures[0]
  curve_lang1 = dataset_mlp_curvatures[1]
  curve_lang2 = dataset_mlp_curvatures[2]
  curve_lang3 = dataset_mlp_curvatures[3]

  plt.scatter(x, curve_lang0 + curve_lang1 + curve_lang2 + curve_lang3, c="blue", marker="^", label="MLPs, Language")

  curve_lang0 = dataset_attn_curvatures[0]
  curve_lang1 = dataset_attn_curvatures[1]
  curve_lang2 = dataset_attn_curvatures[2]
  curve_lang3 = dataset_attn_curvatures[3]

  plt.scatter(x, curve_lang0 + curve_lang1 + curve_lang2 + curve_lang3, c="red", marker="^", label="ATTNs, Language")

  plt.legend()
  plt.title("Global Curvature of the Residual Stream")
  plt.ylabel("Global Curvature")
  plt.xlabel("Number of ICL Examples")
  plt.xticks([0, 1, 2, 3])
  if save_fig and save_path:
    plt.savefig(save_path)
  if show:
    plt.show()
  plt.close()