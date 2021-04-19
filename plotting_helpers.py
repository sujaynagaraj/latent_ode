import matplotlib.pyplot as plt

def to_numpy(x):
		return x.detach().cpu().numpy()

def make_quick_plot(batch_dict, example, extra_str=""):

    plot_dict = {x: to_numpy(batch_dict[x]) for x in ('observed_data', 'observed_tp', 'data_to_predict', 'tp_to_predict', 'observed_mask')}

    plt.figure()
    plt.scatter(plot_dict["observed_tp"], plot_dict["observed_data"][example, :, 0])
    plt.title(f"Observed TP vs Observed Data mode: {batch_dict['mode']}")
    plt.savefig(f"observed_tp_vs_observed_data_{extra_str}_{batch_dict['mode']}_{example}.png")
    

    plt.figure()
    plt.scatter(plot_dict["tp_to_predict"], plot_dict["data_to_predict"][example, :, 0])
    plt.title(f"TP to predict vs Data to predict mode: {batch_dict['mode']}")
    plt.savefig(f"tp_to_predict_vs_data_to_predict_{extra_str}_{batch_dict['mode']}_{example}.png")

    plt.figure()
    plt.scatter(plot_dict["tp_to_predict"], plot_dict["observed_mask"][example, :])
    plt.title(f"TP to predict vs Data to predict amsk: {batch_dict['mode']}")
    plt.savefig(f"tp_to_predict_vs_data_to_predict__mask_{extra_str}_{batch_dict['mode']}_{example}.png")
